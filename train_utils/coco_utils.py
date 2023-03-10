import torch
import torchvision
import torch.utils.data
from pycocotools.coco import COCO


def coco_remove_images_without_annotations(dataset, ids):
    """
    删除coco数据集中没有目标，或者目标面积非常小的数据
    refer to:
    https://github.com/pytorch/vision/blob/master/references/detection/coco_utils.py
    :param dataset:
    :param cat_list:
    :return:
    """
    def _has_only_empty_bbox(anno):
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False

        return True

    valid_ids = []
    for ds_idx, img_id in enumerate(ids):
        ann_ids = dataset.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.loadAnns(ann_ids)

        if _has_valid_annotation(anno):
            valid_ids.append(img_id)

    return valid_ids


def convert_to_coco_api(self):
    coco_ds = COCO()
    # annotation IDs need to start at 1, not 0, see torchvision issue #1530
    ann_id = 1
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(self)):
        targets, h, w = self.get_annotations(img_idx)
        img_id = targets["image_id"].item()
        img_dict = {"id": img_id,
                    "height": h,
                    "width": w}
        dataset["images"].append(img_dict)
        bboxes = targets["boxes"].clone()
        # convert (x_min, ymin, xmax, ymax) to (xmin, ymin, w, h)
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        # if "masks" in targets:
        #     masks = targets["masks"]
        #     # make masks Fortran contiguous for coco_mask
        #     masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {"image_id": img_id,
                   "bbox": bboxes[i],
                   "category_id": labels[i],
                   "area": areas[i],
                   "iscrowd": iscrowd[i],
                   "id": ann_id}
            categories.add(labels[i])
            # if "masks" in targets:
            #     ann["segmentation"] = coco_mask.encode(masks[i].numpy())
            dataset["annotations"].append(ann)
            ann_id += 1
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    return convert_to_coco_api(dataset)
