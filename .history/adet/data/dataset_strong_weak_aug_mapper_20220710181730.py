# Copyright (c) Tencent, Inc. and its affiliates.
# Modified by Tao Wu from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
import copy
import logging
import random
from itertools import cycle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

from pycocotools import mask as coco_mask
import pdb

__all__ = ["COCOInstancePairOverlapDatasetMapper"]


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))
    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


def build_crop_gen(cfg, is_train):
    if cfg.INPUT.CROP.ENABLED and is_train:
        crop_gen = [
            T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
            T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
        ]
    else:
        crop_gen = None
    return crop_gen


def build_strong_transform_gen(cfg, is_train):
    if is_train:
        tfms = [
            T.RandomBrightness(intensity_min=0.5, intensity_max=2.),
            T.RandomLighting(scale=0.2),
            T.RandomSaturation(intensity_min=0.5, intensity_max=2.),
            T.RandomContrast(intensity_min=0.3, intensity_max=20),
        ]
    return tfms


def show_dataset_dict(image, dataset_dict):
    COLORS = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255),
              (255, 128, 0), (255, 0, 128), (0, 255, 128), (0, 128, 255), (128, 0, 255), (128, 255, 0)]
    ncolor = len(COLORS)
    for i, ann in enumerate(dataset_dict['annotations']):
        x, y, w, h = ann['bbox']
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        cv2.rectangle(image, (x1, y1), (x2, y2), COLORS[i % ncolor], 1)
        polygons = [np.array(poly, dtype=np.int).reshape(-1, 2) for poly in ann['segmentation']]
        cv2.polylines(image, polygons, True, COLORS[i % ncolor], 1)
    plt.imshow(image)
    plt.show()


# This is specifically designed for the COCO dataset.
class COCOInstancePairOverlapDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
            self,
            is_train=True,
            *,
            tfm_gens,
            crop_gens,
            strong_gens,
            image_format,
            input_cfg,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        self.crop_gens = crop_gens
        self.strong_gens = strong_gens

        self.img_format = image_format
        self.is_train = is_train
        self.input_cfg = input_cfg

    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)
        crop_gens = build_crop_gen(cfg, is_train)
        strong_gens = build_strong_transform_gen(cfg, is_train)

        ret = {
            "is_train"    : is_train,
            "tfm_gens"    : tfm_gens,
            "crop_gens"   : crop_gens,
            "strong_gens" : strong_gens,
            "image_format": cfg.INPUT.FORMAT,
            "input_cfg"   : {'min_size'    : cfg.INPUT.MIN_SIZE_TRAIN,
                             'max_size'    : cfg.INPUT.MAX_SIZE_TRAIN,
                             'sample_style': cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
                             }
        }
        return ret

    def process_data(self, dataset_dict, image, transforms):

        result_dict = dataset_dict

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # therefore it's important to use torch.Tensor.
        result_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Implement additional transformations if you have other types of data
        annos = []
        for obj in dataset_dict['annotations']:
            if obj.get("iscrowd", 0) == 0:
                annos.append(utils.transform_instance_annotations(obj, transforms, image_shape))

        # NOTE: does not support BitMask due to augmentation
        # Current BitMask cannot handle empty objects
        instances = utils.annotations_to_instances(annos, image_shape)

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if hasattr(instances, 'gt_masks'):
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        # Need to filter empty instances first (due to augmentation)
        instances = utils.filter_empty_instances(instances)

        # Generate masks from polygon
        h, w = instances.image_size
        if hasattr(instances, 'gt_masks'):
            gt_masks = instances.gt_masks
            gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            instances.gt_masks = gt_masks
        else:
            instances.gt_masks = torch.zeros((0, *image_shape))

        result_dict["instances"] = instances

        return result_dict

    def locate_region(self, overlap_bbox, image_size):
        """compute location of cropped region

        Args:
            overlap_bbox (tuple or list): location of overlap region, with format (x_min, y_min, x_max, y_max)
            image_size (tuple): size of image

        Returns:
            location of cropped region, overlap region in cropped region
        """
        ox1, oy1, ox2, oy2 = overlap_bbox
        width, height = image_size
        cw, ch = ox2 - ox1, oy2 - oy1

        # location of cropped region in original image
        rx1 = random.randint(0, ox1)
        ry1 = random.randint(0, oy1)
        rx2 = random.randint(ox2, width)
        ry2 = random.randint(oy2, height)

        # location of overlap region in cropped region
        px1 = ox1 - rx1
        py1 = oy1 - ry1
        px2 = px1 + cw
        py2 = py1 + ch
        return (rx1, ry1, rx2, ry2), (px1, py1, px2, py2)

    def update_annotation(self, dataset_dict, crop_box, overlap_box):
        rx1, ry1, rx2, ry2 = crop_box
        rw, rh = rx2 - rx1, ry2 - ry1
        dataset_dict['height'] = rh
        dataset_dict['width'] = rw
        dataset_dict['overlap_bbox'] = overlap_box

        annos = []
        for ann in dataset_dict['annotations']:
            x1, y1, w, h = ann['bbox']
            x1 = max(x1 - rx1, 0)
            y1 = max(y1 - ry1, 0)
            w = min(w, rw - x1)
            h = min(h, rh - y1)
            if w < 1 or h < 1:
                continue
            ann['bbox'] = [x1, y1, w, h]

            if not ann['iscrowd']:
                poly = []
                for p in ann['segmentation']:
                    p = np.array(p).reshape(-1, 2)
                    p[:, 0] = np.minimum(np.maximum(0, p[:, 0] - rx1), rw)
                    p[:, 1] = np.minimum(np.maximum(0, p[:, 1] - ry1), rh)
                    poly.append(p.reshape(1, -1).tolist())
                ann['segmentation'] = poly

            annos.append(ann)
        dataset_dict['annotations'] = annos

    def get_status(self, transforms):
        status = dict(hflip=False, align_size=None)
        for tran in transforms:
            if tran.__class__.__name__ == 'HFlipTransform':
                status['hflip'] = True
            elif tran.__class__.__name__ == 'ResizeTransform':
                status['align_size'] = dict(h=tran.h, w=tran.w)
        return status

    def weak_transform(self, dataset_dict, image, transforms):
        """weak augmentation"""
        # height, width = image.shape[:2]
        # crop_box, ol_box = self.locate_region(overlap_bbox, (width, height))
        # rx1, ry1, rx2, ry2 = crop_box
        # image = image[ry1:ry2, rx1:rx2]
        # self.update_annotation(dataset_dict, crop_box, ol_box)  # update dataset_dict

        # show_dataset_dict(image, dataset_dict)

        # augs = [
        #     T.ResizeShortestEdge(self.input_cfg['min_size'],
        #                          self.input_cfg['max_size'],
        #                          self.input_cfg['sample_style']),
        # ]
        
        # image, transforms = T.apply_transform_gens(augs, image)
        dataset_dict['status'] = self.get_status(transforms)
        return self.process_data(dataset_dict, image, transforms)

    def strong_transform(self, dataset_dict, image, transforms):
        """strong transforms, without geometric transform"""
        # height, width = image.shape[:2]
        # crop_box, ol_box = self.locate_region(overlap_bbox, (width, height))
        # rx1, ry1, rx2, ry2 = crop_box
        # image = image[ry1:ry2, rx1:rx2]
        # self.update_annotation(dataset_dict, crop_box, ol_box)  # update dataset_dict
        # augs = [
        #     T.ResizeShortestEdge(self.input_cfg['min_size'],
        #                          self.input_cfg['max_size'],
        #                          self.input_cfg['sample_style']),
        #     # T.RandomFlip(),
        #     *random.sample(self.strong_gens, 2)
        # ]
        augs = [*random.sample(self.strong_gens, 2)]
        image, transforms_s = T.apply_transform_gens(augs, image)
        transforms = transforms + transforms_s
        dataset_dict['status'] = self.get_status(transforms)
        return self.process_data(dataset_dict, image, transforms)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # pdb.set_trace()
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        crop_ratio = [0.3, 0.3]
        height, width = image.shape[:2]
        crop_ratio = np.asarray(crop_ratio, dtype=np.float32)
        ch, cw = crop_ratio + np.random.rand(2) * (1 - crop_ratio)
        ch, cw = int(height * ch + 0.5), int(width * cw + 0.5)

        # location of overlap region in original image
        ox1 = random.randint(0, width - cw)
        oy1 = random.randint(0, height - ch)
        ox2 = ox1 + cw
        oy2 = oy1 + ch

        augs = [
            T.ResizeShortestEdge(self.input_cfg['min_size'],
                                 self.input_cfg['max_size'],
                                 self.input_cfg['sample_style']),
        ]
        image, transforms = T.apply_transform_gens(augs, image)
        result_weak = self.weak_transform(copy.deepcopy(dataset_dict), image, transforms)
        result_strong = self.strong_transform(copy.deepcopy(dataset_dict), image, transforms)

        # w1 = result_weak['overlap_bbox'][2] - result_weak['overlap_bbox'][0]
        # h1 = result_weak['overlap_bbox'][3] - result_weak['overlap_bbox'][1]
        # w2 = result_strong['overlap_bbox'][2] - result_strong['overlap_bbox'][0]
        # h2 = result_strong['overlap_bbox'][3] - result_strong['overlap_bbox'][1]
        # assert w1 == w2 and h1 == h2

        return result_weak, result_strong