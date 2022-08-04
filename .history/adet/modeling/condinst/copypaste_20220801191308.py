# from numpy import random
import random
import numpy as np
from skimage.filters import gaussian
import torch
import torch.nn.functional as F
from detectron2.structures import (
    Boxes,
    Instances,
)
import pdb

def _copypaste_transform(images_norm, gt_instances, paste_data):
    renewed_instances = []
    original_imgs = []
    for i in range(len(images_norm)):
        # get data
        original_img = images_norm.tensor[i]
        _, orig_W, orig_H = original_img.shape
        # original_gt_bboxes = original_data['gt_bboxes']
        original_gt_labels = gt_instances[i].gt_classes
        add_bitmasks_from_boxes(gt_instances[i], orig_W, orig_H)
        original_mask = gt_instances[i].gt_masks_full


        index = random.randint(0, len(paste_data)-1)
        paste_img = paste_data[index]['image'].to(original_img.device)
        paste_gt_bboxes = paste_data[index]['gt_boxes']
        paste_gt_labels = paste_data[index]['gt_classes']
        paste_mask = paste_data[index]['paste_mask'].to(original_img.device)
        # paste_weight = paste_data[index]['paste_weight'].to(original_img.device)

        # choice paste bbox mask
        n_objects = len(paste_gt_bboxes.tensor)

        n_select = int(n_objects * 0.5)
        n_select = max(1, n_select)
        n_select = min(n_select, 3)

        paste_indexs = random.choices(range(0, n_objects), weights=paste_data[index]['score']**3, k=n_select) 
        # paste_indexs = random.choice(range(0, n_objects), size=n_select, replace=False)

        paste_gt_bboxes = paste_gt_bboxes[paste_indexs]
        paste_gt_labels = paste_gt_labels[paste_indexs]
        
        # pdb.set_trace()
        paste_mask = paste_mask[paste_indexs]

        _, past_W, past_H = paste_img.shape

        max_W, max_H = max(orig_W, past_W), max(orig_H, past_H)
        original_img = F.pad(original_img, (0, max_H-orig_H, 0, max_W-orig_W, 0, 0), 'constant')
        original_mask = F.pad(original_mask, (0, max_H-orig_H, 0, max_W-orig_W, 0, 0), 'constant')
        paste_img = F.pad(paste_img, (0, max_H-past_H, 0, max_W-past_W, 0, 0), 'constant')
        paste_mask = F.pad(paste_mask, (0, max_H-past_H, 0, max_W-past_W, 0, 0), 'constant')
      
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
       
        original_img = out_img[:,:orig_W,:orig_H]
        
        # out_mask = BitmapMasks(out_mask, out_img.shape[0], out_img.shape[1])
        results = {}
        results['img_shape'] = (orig_W, orig_H)
        results['gt_bboxes'] = out_bbox.to(original_img.device)
        results['gt_labels'] = torch.tensor(out_labels, dtype=torch.int64).to(original_img.device)
        results['gt_masks'] = out_mask[:, :orig_W, :orig_H]
        # results['paste_weight'] = F.pad(paste_weight[paste_indexs], \
        #     (0, max_H-past_H, 0, max_W-past_W, len(out_mask)-len(paste_weight[paste_indexs]), 0), 'constant')[:, :orig_W, :orig_H]

        instances = annotations_to_instances(results)
        indicator = torch.zeros(len(keep_index))
        indicator[-n_select:] = 1
        instances.paste_indicator = indicator.to(original_img.device)
        
        original_imgs.append(original_img)
        renewed_instances.append(instances)

    return original_imgs, renewed_instances


def annotations_to_instances(results):
    target = Instances(results['img_shape'])
    target.gt_boxes = results['gt_bboxes']
    target.gt_classes = results['gt_labels']
    target.gt_bitmasks_full = results['gt_masks']
    stride = 4
    start = int(stride // 2)
    # target.gt_weight = results['paste_weight'][:, start::stride, start::stride]
    target.gt_bitmasks = target.gt_bitmasks_full[:, start::stride, start::stride]
    return target


def add_bitmasks_from_boxes(gt_instance, im_h, im_w):
    stride = 4 
    start = int(stride // 2)
    per_im_boxes = gt_instance.gt_boxes.tensor
    per_im_bitmasks = []
    per_im_bitmasks_full = []
    for per_box in per_im_boxes:
        bitmask_full = torch.zeros((im_h, im_w)).to(gt_instance.gt_classes.device).float()
        bitmask_full[int(per_box[1]):int(per_box[3] + 1), int(per_box[0]):int(per_box[2] + 1)] = 1.0

        bitmask = bitmask_full[start::stride, start::stride]

        assert bitmask.size(0) * stride == im_h
        assert bitmask.size(1) * stride == im_w

        per_im_bitmasks.append(bitmask)
        per_im_bitmasks_full.append(bitmask_full)
    gt_instance.gt_masks = torch.stack(per_im_bitmasks, dim=0)
    gt_instance.gt_masks_full = torch.stack(per_im_bitmasks_full, dim=0)  


def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)
        img_dtype = img.dtype
        alpha = alpha[None, ...]

        img = paste_img * alpha + img * (~alpha)
        img = img.to(img_dtype)
    return img

def masks_copy_paste(masks, paste_masks, alpha):
    if alpha is not None:
        # eliminate pixels that will be pasted over
        masks = [(mask.int() & (mask.int() ^ alpha)) for mask in masks]
        masks.extend(paste_masks)

    return torch.stack(masks)

def extract_bboxes(out_masks):
    bboxes = []
    keep_index = []
    for i, mask in enumerate(out_masks):
        yindices = torch.where(torch.any(mask.to(torch.uint8), axis=0))[0]
        xindices = torch.where(torch.any(mask.to(torch.uint8), axis=1))[0]
        if yindices.shape[0]:
            y1, y2 = yindices[[0, -1]]
            x1, x2 = xindices[[0, -1]]
            y2 += 1
            x2 += 1

            bboxes.append((y1, x1, y2, x2))
            keep_index.append(i)

    return Boxes(bboxes), keep_index