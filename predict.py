import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_objs


def create_model(num_classes, loss_fn, focal, cbam, double_fusion, anchor, val):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))
    #
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    if anchor == 64:
        anchor_sizes = ((32,), (64,), (128,), (256,))  # 32 512  # 32 512
        aspect_ratios = ((0.2, 0.33, 0.5, 1.0, 2.0, 3.0, 5.0),) * len(anchor_sizes)
    elif anchor == 32:
        anchor_sizes = ((32,), (64,), (128,), (256,))  # 32 512  # 32 512
        aspect_ratios = ((0.33, 0.5, 1.0, 2.0, 3.0),) * len(anchor_sizes)
    elif anchor == 0.33:
        anchor_sizes = ((32,), (64,), (128,), (256,))  # 32 512  # 32 512
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    elif anchor == 0.2:
        anchor_sizes = ((64,), (128,), (256,))  # 32 512  # 32 512
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorsGenerator(anchor_sizes, aspect_ratios)
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, rpn_anchor_generator=anchor_generator,
                       num_classes=num_classes, loss_fn=loss_fn, focal=focal, cbam=cbam, double_fusion=double_fusion, val=val)
    # model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    num_classes = 2
    loss_fn = "l1"
    focal = False
    cbam = False
    double_fusion = False
    anchor = 64

    # create model
    model = create_model(num_classes, loss_fn, focal, cbam, double_fusion, anchor, val=False)

    # load train weights
    weights_path = "./save_weights/origin.pth"
    assert os.path.exists(weights_path), "{} file dose not exist.".format(weights_path)
    weights_dict = torch.load(weights_path, map_location='cpu')

    weights_dict = weights_dict["model"] if "model" in weights_dict else weights_dict
    model.load_state_dict(weights_dict, strict=False)
    model.to(device)

    # read class_indict
    label_json_path = './coco_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    with open(label_json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # load image
    original_img = Image.open("./data/test/infer/037_03_contrast85_2_c.jpg")

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        predictions = model(img.to(device))[0]
        t_end = time_synchronized()
        print("inference+NMS time: {}".format(t_end - t_start))

        prediction = predictions[0]
        predict_boxes = prediction["boxes"].to("cpu").numpy()
        predict_classes = prediction["labels"].to("cpu").numpy()
        predict_scores = prediction["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        plot_img = draw_objs(original_img,
                             predict_boxes,
                             predict_classes,
                             predict_scores,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)
        plt.imshow(plot_img)
        plt.show()
        # 保存预测的图片结果
        plot_img.save("./data/test/infer/iou_result.jpg")


if __name__ == '__main__':
    main()
