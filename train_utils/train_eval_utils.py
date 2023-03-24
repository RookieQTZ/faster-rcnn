import math
import sys
import time

import matplotlib.pyplot as plt
import torch
import numpy as np

from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils
import plot_curve


def train_one_epoch(model, optimizer, data_loader, weighted_loss_func, device, epoch, last_loss,
                    print_freq=50, warmup=False, scaler=None, adaptive_weight=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses

    for i, [org_ul_images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = [org_ul_image[0] for org_ul_image in org_ul_images]
        # ul_images = [org_ul_image[1] for org_ul_image in org_ul_images]

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            # 为分类损失多加点损失权重？
            if not adaptive_weight:
                losses = sum(loss for loss in loss_dict.values())
            else:
                # 自适应多任务损失权重
                cur_loss = [loss for loss in loss_dict.values()]
                weight = get_losses_weights(cur_loss)
                new_losses = [loss * w for loss, w in zip(cur_loss, weight)]  # new_losses: [0.6919, 0.5297, 1.8270, 0.8757])
                losses = sum(loss for loss in new_losses)  # loss: 3.9243
                if i == 0:
                    print("Epoch: [" + str(epoch) + "]  weight: " + str([str(w.item()) for w in weight]))

            # losses, sigma = weighted_loss_func(*[loss for loss in loss_dict.values()])

        # 每个epoch开始时打印一次sigma
        # if i == 0:
        #     print("Epoch: [" + str(epoch) + "]  sigma: " + sigma)

        # reduce losses over all GPUs for logging purpose
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()
        # 记录训练损失
        mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

        if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, loss_dict, now_lr, weight


@torch.no_grad()
def evaluate(model, data_loader, epoch, last_epoch, viz, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    # 将自己的dataset转换成coco能识别的形式，以便后续进行eval
    # voc数据集才需要转换，coco数据集格式已定义好并存在dataloader中
    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # coco = EvalCOCOMetric(data_loader.dataset.coco, iou_type="bbox", results_file_name="det_results.json")
    coco = data_loader.dataset.coco
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    mloss = torch.zeros(1).to(device)  # mean losses
    for i, (images, targets) in enumerate(metric_logger.log_every(data_loader, 100, header)):
        image = list(imgs[0].to(device) for imgs in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs, loss = model(image, targets)

        losses = sum(loss for loss in loss.values())
        # 记录训练损失
        mloss = (mloss * i + losses) / (i + 1)  # update mean losses

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    print("val loss:", mloss.item())
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    # coco_evaluator找到数据，绘制pr曲线
    if epoch == last_epoch:
        # evaluate_predictions_on_coco(coco_evaluator.coco_eval[iou_types[0]])
        plot_curve.visdom_pr(viz, coco_evaluator.coco_eval[iou_types[0]])

    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types


def get_losses_weights(loss_list: [list, np.ndarray, torch.Tensor]):
    if type(loss_list) != torch.Tensor:
        loss_list = torch.tensor(loss_list)
    weights = torch.div(loss_list, torch.sum(loss_list)) * loss_list.shape[0]
    # weights = torch.round(weights, decimals=4)
    return weights

# losses = torch.tensor([0.8, 0.7, 1.3, 0.9])	# old_loss = torch.sum(losses) = 3.7
# loss_w = get_losses_weights(losses)			# loss_w: [0.8649, 0.7568, 1.4054, 0.9730]
# new_losses = losses * loss_w				# new_losses: [0.6919, 0.5297, 1.8270, 0.8757])
# loss = torch.sum(new_losses)				# loss: 3.9243
