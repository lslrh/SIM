import random
import pdb


def _copypaste_transform(images_norm, gt_instances, paste_data):
    pdb.set_trace()
    for i in range(len(images_norm)):
        # get data
        original_img = images_norm.tensor[i]
        # original_gt_bboxes = original_data['gt_bboxes']
        original_gt_labels = gt_instances[i].gt_classes
        original_mask = gt_instances[i].gt_bitmasks

        paste_img = paste_data[i]['image']
        paste_gt_bboxes = paste_data[i]['gt_boxes']
        paste_gt_labels = paste_data[i]['gt_labels']
        paste_mask = paste_data['paste_mask']

        # choice paste bbox mask
        n_objects = len(paste_gt_bboxes)


        n_select = int(n_objects * 0.25)
        n_select = max(1, n_select)
        n_select = min(n_select, 3)

        paste_indexs = random.choice(range(0, n_objects), size=n_select, replace=False)

        paste_gt_bboxes = paste_gt_bboxes[paste_indexs]
        paste_gt_labels = paste_gt_labels[paste_indexs]
        paste_mask = paste_mask[paste_indexs]


        alpha = paste_mask[0] > 0
        for mask in paste_mask[1:]:
            alpha += mask > 0


        # 执行copy-paste
        out_img = self.image_copy_paste(original_img, paste_img, alpha, self.blend, self.sigma)
        out_mask = self.masks_copy_paste(original_mask, paste_mask, alpha)
        out_bbox, keep_index = self.extract_bboxes(out_mask)
        out_labels = np.append(original_gt_labels, paste_gt_labels)

        out_mask = out_mask[keep_index]
        out_labels = out_labels[keep_index]


        out_mask = BitmapMasks(out_mask, out_img.shape[0], out_img.shape[1])

        results['img'] = out_img
        results['img_shape'] = out_img.shape
        results['ori_shape'] = out_img.shape
        results['gt_bboxes'] = out_bbox
        results['gt_labels'] = out_labels
        results['gt_masks'] = out_mask

    # import pdb
    # pdb.set_trace()

    return results



def image_copy_paste(img, paste_img, alpha, blend=True, sigma=1):
    if alpha is not None:
        if blend:
            alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

        img_dtype = img.dtype
        alpha = alpha[..., None]
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