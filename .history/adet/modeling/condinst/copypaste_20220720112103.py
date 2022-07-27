from numpy import random
import numpy as np
from skimage.filters import gaussian
import torch
from detectron2.structures import (
    Boxes,
    Instances,
)
import pdb

def _copypaste_transform(images_norm, gt_instances, paste_data):
    renewed_instances = []
    for i in range(len(images_norm)):
        # get data
        original_img = images_norm.tensor[i].cpu().numpy()
        _, orig_W, orig_H = original_img.shape
        # original_gt_bboxes = original_data['gt_bboxes']
        original_gt_labels = gt_instances[i].gt_classes
        add_bitmasks_from_boxes(gt_instances[i], orig_W, orig_H)
        original_mask = gt_instances[i].gt_bitmasks_full.numpy()

        index = np.random.randint(0, 49)
        paste_img = paste_data[index]['image']
        paste_gt_bboxes = paste_data[index]['gt_boxes']
        paste_gt_labels = paste_data[index]['gt_classes']
        paste_mask = paste_data[index]['paste_mask']

        # choice paste bbox mask
        n_objects = len(paste_gt_bboxes)

        n_select = int(n_objects * 0.25)
        n_select = max(1, n_select)
        n_select = min(n_select, 3)

        paste_indexs = random.choice(range(0, n_objects), size=n_select, replace=False)

        paste_gt_bboxes = paste_gt_bboxes[paste_indexs]
        paste_gt_labels = paste_gt_labels[paste_indexs]
        paste_mask = paste_mask[paste_indexs]

        _, past_W, past_H = paste_img.shape

        max_W, max_H = max(orig_W, past_W), max(orig_H, past_H)
        original_img = np.pad(original_img, ((0, 0), (0, max_W-orig_W), (0, max_H-orig_H)), 'constant')
        original_mask = np.pad(original_mask, ((0, 0), (0, max_W-orig_W), (0, max_H-orig_H)), 'constant')
        paste_img = np.pad(paste_img, ((0, 0), (0, max_W-past_W), (0, max_H-past_H)), 'constant')
        paste_mask = np.pad(paste_mask, ((0, 0), (0, max_W-past_W), (0, max_H-past_H)), 'constant')
      
        alpha = paste_mask[0] > 0
        for mask in paste_mask[1:]:
            alpha += mask > 0

        # 执行copy-paste
        out_img = image_copy_paste(original_img, paste_img, alpha, blend=False, sigma=3)
        out_mask = masks_copy_paste(original_mask, paste_mask, alpha)
        out_bbox, keep_index = extract_bboxes(out_mask)
        out_labels = np.append(original_gt_labels.tolist(), paste_gt_labels.tolist())

        out_mask = out_mask[keep_index]
        out_labels = out_labels[keep_index]

        original_img = torch.from_numpy(out_img[:,:orig_W,:orig_H]).to(images_norm.device)
        
        # out_mask = BitmapMasks(out_mask, out_img.shape[0], out_img.shape[1])
        results = {}
        results['img_shape'] = (orig_W, orig_H)
        results['gt_bboxes'] = out_bbox
        results['gt_labels'] = out_labels
        results['gt_masks'] = out_mask[:, :orig_W, :orig_H]

        instances = annotations_to_instances(results)
        pdb.set_trace()
        indicator = torch.zeros(len(keep_index))
        instances.paste_indicator = indicator 
        
        renewed_instances.append(instances)

    return original_img, renewed_instances


def annotations_to_instances(results):
    target = Instances(results['img_shape'])
    target.gt_boxes = Boxes(results['gt_bboxes'])
    target.gt_classes = torch.tensor(results['gt_labels'], dtype=torch.int64)
    target.gt_bitmasks_full = torch.tensor(results['gt_masks'])
    stride = 4
    start = int(stride // 2)
    target.gt_bitmasks = target.gt_bitmasks_full[:, start::stride, start::stride]
    return target


def add_bitmasks_from_boxes(gt_instance, im_h, im_w):
    stride = 4 
    start = int(stride // 2)
    per_im_boxes = gt_instance.gt_boxes.tensor
    per_im_bitmasks = []
    per_im_bitmasks_full = []
    for per_box in per_im_boxes:
        bitmask_full = torch.zeros((im_h, im_w)).float()
        bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

        bitmask = bitmask_full[start::stride, start::stride]

        assert bitmask.size(0) * stride == im_h
        assert bitmask.size(1) * stride == im_w

        per_im_bitmasks.append(bitmask)
        per_im_bitmasks_full.append(bitmask_full)
    gt_instance.gt_bitmasks = torch.stack(per_im_bitmasks, dim=0)
    gt_instance.gt_bitmasks_full = torch.stack(per_im_bitmasks_full, dim=0)  


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[None, ...]

        img = paste_img * alpha + img * (1 - alpha)
        img = img.astype(img_dtype)

    return img

def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [
            np.logical_and(mask, np.logical_xor(mask, alpha)).astype(np.uint8) for mask in masks
        ]
        masks.extend(paste_masks)

    return np.array(masks)

def extract_bboxes(out_masks):
    bboxes = []
    keep_index = []
    for i, mask in enumerate(out_masks):
        yindices = np.where(np.any(mask, axis=0))[0]
        xindices = np.where(np.any(mask, axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1

            bboxes.append((y1, x1, y2, x2))
            keep_index.append(i)

    return np.array(bboxes).astype(np.float32), np.array(keep_index).astype(np.uint8)