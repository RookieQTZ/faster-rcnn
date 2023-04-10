import os
import datetime

import torch

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from my_dataset_coco import CocoDetection
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils
from network_files import multitask_loss
import plot_curve
from train_utils import loss_weight


def create_model(args, num_classes, load_pretrain_weights=True):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    # resnet50 imagenet weights url: https://download.pytorch.org/models/resnet50-0676ba61.pth
    backbone = resnet50_fpn_backbone(pretrain_path=args.res_pretrain_path,
                                     norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91, loss_fn=args.loss_fn, focal=args.focal, cbam=args.cbam, double_fusion=args.double_fusion, )

    if load_pretrain_weights:
        # 载入预训练模型权重
        # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
        weights_dict = torch.load(args.rcnn_pretrain_path, map_location='cpu')
        weights_dict.pop('rpn.head.cls_logits.weight')
        weights_dict.pop('rpn.head.cls_logits.bias')
        weights_dict.pop('rpn.head.bbox_pred.weight')
        weights_dict.pop('rpn.head.bbox_pred.bias')
        missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
        if len(missing_keys) != 0 or len(unexpected_keys) != 0:
            print("missing_keys: ", missing_keys)
            print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    # results_file = "./save_weights/results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    results_file = "./save_weights/results.txt"
    visdom_file = "./save_weights/visdom.log"
    sigma_file = "./save_weights/sigma.txt"
    loss_weight_file = "./save_weights/loss_weight.txt"

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    data_root = args.data_path

    # load train data set
    # coco2017 -> annotations -> instances_train2017.json
    train_dataset = CocoDetection(data_root, "train", data_transform["train"])
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # coco2017 -> annotations -> instances_train2017.json
    val_dataset = CocoDetection(data_root, "val", data_transform["val"])
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create model num_classes equal background + 20 classes
    model = create_model(args, num_classes=args.num_classes + 1)
    print("using focal: " + str(args.focal))
    print("using adaptive_weight: " + str(args.adaptive_weight))
    # print(model)

    model.to(device)

    # define optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,
                                lr=args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    ########### UncertaintyLoss ############
    weighted_loss_func = multitask_loss.UncertaintyLoss(4)
    weighted_loss_func.to(device)
    # params = filter(lambda x: x.requires_grad, list(model.parameters()) + list(weighted_loss_func.parameters()))
    # optimizer = torch.optim.SGD(params,
    #                             lr=args.lr,
    #                             momentum=args.momentum,
    #                             weight_decay=args.weight_decay)
    ########### UncertaintyLoss ############

    scaler = torch.cuda.amp.GradScaler() if args.amp else None

    # learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   # step_size=3,  # epoch=20
                                                   # gamma=0.33)
                                                   step_size=2,  # epoch=10
                                                   gamma=0.5)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    # viz = plot_curve.create_visdom(visdom_file)
    if args.resume != "":
        # plot_curve.load_visdom(viz, visdom_file)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        args.start_epoch = checkpoint['epoch'] + 1
        if args.amp and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        print("the training process from epoch{}...".format(args.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch, printing every 10 iterations
        mean_loss, loss_dict, lr = utils.train_one_epoch(model, optimizer, train_data_loader, weighted_loss_func,
                                                         device=device, epoch=epoch,
                                                         print_freq=50, warmup=True,
                                                         scaler=scaler, adaptive_weight=args.adaptive_weight)

        # 保存last_loss权重
        # loss_weight.save(weight, loss_weight_file)

        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()

        # evaluate on the test dataset
        # todo: 最后一个epoch，才绘制pr曲线
        # todo: 拿到所有ap、ar信息，分别保存
        # todo: 所有损失
        coco_info = utils.evaluate(model, val_data_set_loader, epoch, args.epochs - 1, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [f"{i:.4f}" for i in coco_info + [mean_loss.item()]] + [f"{lr:.6f}"]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")

        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        if args.amp:
            save_files["scaler"] = scaler.state_dict()
        torch.save(save_files, "./save_weights/resNetFpn-model-{}.pth".format(epoch))

        # 损失曲线
        # plot_curve.visdom_draw(viz, mean_loss, epoch, title='Loss', ylabel='loss')
        # plot_curve.visdom_draw(viz, loss_dict['loss_classifier'], epoch, title='Loss Classifier', ylabel='loss')
        # plot_curve.visdom_draw(viz, loss_dict['loss_box_reg'], epoch, title='Loss Box Reg', ylabel='loss')
        # plot_curve.visdom_draw(viz, loss_dict['loss_objectness'], epoch, title='Loss Objectness', ylabel='loss')
        # plot_curve.visdom_draw(viz, loss_dict['loss_rpn_box_reg'], epoch, title='Loss Rpn Box Reg', ylabel='loss')
        # # ap ar
        # plot_curve.visdom_draw(viz, coco_info[0], epoch, title='mAP', ylabel='mAP')
        # plot_curve.visdom_draw(viz, coco_info[3], epoch, title='AP Small', ylabel='AP')
        # plot_curve.visdom_draw(viz, coco_info[4], epoch, title='AP Medium', ylabel='AP')
        # plot_curve.visdom_draw(viz, coco_info[5], epoch, title='AP Large', ylabel='AP')
        # plot_curve.visdom_draw(viz, coco_info[9], epoch, title='AR Small', ylabel='AR')
        # plot_curve.visdom_draw(viz, coco_info[10], epoch, title='AR Medium', ylabel='AR')
        # plot_curve.visdom_draw(viz, coco_info[11], epoch, title='AR Large', ylabel='AR')
        # pr

    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse

    def str2bool(s):
        return True if s.lower() == "true" else False

    parser = argparse.ArgumentParser(
        description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data_path', default='./data/test', help='dataset')
    # resnet预训练模型地址
    parser.add_argument('--res_pretrain_path', default='resnet50.pth', help='resnet50 pretrained path')
    # faster rcnn fpn预训练模型地址
    parser.add_argument('--rcnn_pretrain_path', default='fasterrcnn_resnet50_fpn_coco.pth', help='rcnn pretrained path')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--num_classes', default=1, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址
    parser.add_argument('--resume', default='', type=str, help='resume from checkpoint')
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # 学习率
    parser.add_argument('--lr', default=0.005, type=float,
                        help='initial learning rate, 0.02 is the default value for training '
                             'on 8 gpus and 2 images_per_gpu')
    # SGD的momentum参数
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    # SGD的weight_decay参数
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    # 训练的batch size
    parser.add_argument('--batch_size', default=1, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)
    # 是否使用混合精度训练(需要GPU支持混合精度)
    parser.add_argument("--amp", default=False, type=str2bool, help="Use torch.cuda.amp for mixed precision training")
    # parser.add_argument("--amp", action="store_true",
    #                     help="Use torch.cuda.amp for mixed precision training")
    # 使用的损失函数
    parser.add_argument("--loss-fn", default='l1', help="loss function to use")
    # 是否使用Focal loss
    parser.add_argument("--focal", default=False, type=str2bool, help="Use focal loss")
    # 是否使用cbam注意力机制
    parser.add_argument("--cbam", default=True, type=str2bool, help="Use cbam attention block")
    # 是否使用双向融合fpn
    parser.add_argument("--double-fusion", default=False, type=str2bool, help="Use double fusion fpn block")
    # 是否使用自适应损失权重
    parser.add_argument("--adaptive_weight", default=False, type=str2bool, help="Use adaptive weight")
    # 分类损失权重系数
    parser.add_argument("--weight", default=1., type=float, help="class task weight")

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
