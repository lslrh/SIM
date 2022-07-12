import torch
from torch.nn import functional as F
from torch import nn
import torch.distributed as dist

from adet.utils.comm import compute_locations, aligned_bilinear
from adet.utils.contrast import l2_normalize, momentum_update
from adet.utils.show import show_feature_map
from adet.utils.sinkhorn import distributed_sinkhorn
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from torch_scatter import scatter_mean
import pdb

def compute_project_term(mask_scores, gt_bitmasks):
    mask_losses_y = dice_coefficient(
        mask_scores.max(dim=2, keepdim=True)[0],
        gt_bitmasks.max(dim=2, keepdim=True)[0]
    )
    mask_losses_x = dice_coefficient(
        mask_scores.max(dim=3, keepdim=True)[0],
        gt_bitmasks.max(dim=3, keepdim=True)[0]
    )
    return (mask_losses_x + mask_losses_y).mean()


def compute_pairwise_term(mask_logits, pairwise_size, pairwise_dilation):
    assert mask_logits.dim() == 4
    log_fg_prob = F.logsigmoid(mask_logits)
    log_bg_prob = F.logsigmoid(-mask_logits)

    from adet.modeling.condinst.condinst import unfold_wo_center
    log_fg_prob_unfold = unfold_wo_center(
        log_fg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )
    log_bg_prob_unfold = unfold_wo_center(
        log_bg_prob, kernel_size=pairwise_size,
        dilation=pairwise_dilation
    )

    # the probability of making the same prediction = p_i * p_j + (1 - p_i) * (1 - p_j)
    # we compute the the probability in log space to avoid numerical instability
    log_same_fg_prob = log_fg_prob[:, :, None] + log_fg_prob_unfold
    log_same_bg_prob = log_bg_prob[:, :, None] + log_bg_prob_unfold

    max_ = torch.max(log_same_fg_prob, log_same_bg_prob)
    log_same_prob = torch.log(
        torch.exp(log_same_fg_prob - max_) +
        torch.exp(log_same_bg_prob - max_)
    ) + max_

    # loss = -log(prob)
    return -log_same_prob[:, 0]


def dice_coefficient(x, target, weights=None):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    if weights is not None:
        weights = weights.reshape(n_inst, -1)
        x = x*weights
        target = target*weights
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    if weights is not None:
        return loss.mean()
    return loss


def mask_focal_loss(x, targets, weights, alpha: float = 0.25, gamma: float = 2):
    ce_loss = F.binary_cross_entropy_with_logits(x, targets, weight=weights, reduction="none")
    p_t = x * targets + (1 - x) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss.sum()/weights.sum() 



def get_feat_similarity(images, kernel_size, dilation):
    assert images.dim() == 4
    from adet.modeling.condinst.condinst import unfold_wo_center

    unfolded_images = unfold_wo_center(
        images, kernel_size=kernel_size, dilation=dilation
    )

    diff = images[:, :, None] - unfolded_images
 
    similarity = torch.exp(-torch.norm(diff, dim=1) * 1)

    return similarity


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS
        self.num_classes = cfg.MODEL.BASIS_MODULE.NUM_CLASSES

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        # boxinst configs
        self.boxinst_enabled = cfg.MODEL.BOXINST.ENABLED
        self.bottom_pixels_removed = cfg.MODEL.BOXINST.BOTTOM_PIXELS_REMOVED
        self.pairwise_size = cfg.MODEL.BOXINST.PAIRWISE.SIZE
        self.pairwise_dilation = cfg.MODEL.BOXINST.PAIRWISE.DILATION
        self.pairwise_color_thresh = cfg.MODEL.BOXINST.PAIRWISE.COLOR_THRESH
        self.pseudo_thresh = cfg.MODEL.BOXINST.PSEUDO_THRESH
        self._warmup_iters = cfg.MODEL.BOXINST.PAIRWISE.WARMUP_ITERS
        self.num_proto = cfg.MODEL.BOXINST.NUM_PROTO 
        self.prototypes = nn.Parameter(torch.zeros(self.num_classes, self.num_proto, self.in_channels), requires_grad=False)

        weight_nums, bias_nums = [], []
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

        self.register_buffer("_iter", torch.zeros([1]))
        trunc_normal_(self.prototypes, std=0.02)


    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            instance_locations = instances.locations
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)

        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits

    def prototype_learning(self, mask_feat, masks, mask_scores, gt_classes, pred_labels, im_inds):
        for b in range(len(mask_feat)):
            gt_classes_unique = torch.unique(gt_classes[b])

            protos = self.prototypes.data.clone()
            for k in gt_classes_unique:
                inds = (im_inds==b) & (pred_labels==k)

                pseudo = mask_scores[inds]
                init_q = masks[inds]
                c_q = mask_feat[b].unsqueeze(0).expand(len(init_q), -1,-1,-1)

                init_q = rearrange(init_q, 'n p h w -> (n h w) p')
                pseudo = rearrange(pseudo, 'n p h w -> (n p h w)')
           
                c_q = rearrange(c_q, 'n h w p -> (n h w) p')

                init_q = init_q[pseudo==1]
                c_q = c_q[pseudo==1]

                if init_q.shape[0] == 0:
                    continue
                q, indexs = distributed_sinkhorn(init_q)

                f = q.transpose(0, 1) @ c_q
                f = F.normalize(f, p=2, dim=-1)
                n = torch.sum(q, dim=0)
                new_value = momentum_update(old_value=protos[k, n!=0, :], new_value=f[n!=0, :], momentum=0.999, debug=False)
                protos[k, n!=0, :] = new_value
            self.prototypes = nn.Parameter(l2_normalize(protos), requires_grad=False)
        
        if dist.is_available() and dist.is_initialized():
            protos = self.prototypes.data.clone()
            dist.all_reduce(protos.div_(dist.get_world_size()))
            self.prototypes = nn.Parameter(protos, requires_grad=False)

    def __call__(self, mask_feats, mask_feats_ema, mask_feat_stride, pred_instances, pred_instances_ema, gt_instances=None):
        if self.training:
            self._iter += 1

            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            losses = {}

            if len(pred_instances) == 0:
                dummy_loss = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
                if not self.boxinst_enabled:
                    losses["loss_mask"] = dummy_loss
                else:
                    losses["loss_prj"] = dummy_loss
                    losses["loss_pairwise"] = dummy_loss
            else:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                mask_scores = mask_logits.sigmoid()
                
                mask_logits_ema = self.mask_heads_forward_with_coords(
                    mask_feats_ema, mask_feat_stride, pred_instances_ema
                )
                mask_scores_ema = mask_logits_ema.sigmoid()
                # show_feature_map(mask_feats[0].detach(), 0)
                # show_feature_map(mask_feats[1].detach(), 1)

                if self.boxinst_enabled:
                    # compute feats similarity 
                    mask_feats_ema = F.interpolate(
                                                mask_feats_ema,
                                                scale_factor=2,
                                                mode="bilinear", align_corners=False)
                    # mask_feat_similarity = get_feat_similarity(mask_feats_ema, self.pairwise_size, self.pairwise_dilation)
                    # mask_feat_similarity_list = []
                    # for i in range(len(gt_instances)):
                    #     mask_feat_similarity_list.append(torch.stack([mask_feat_similarity[i] for _ in range(len(gt_instances[i]))], dim=0))
                    # mask_feat_similarity = torch.cat([x for x in mask_feat_similarity_list])
                    # mask_feat_similarity = mask_feat_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    # box-supervised BoxInst losses
                    image_color_similarity = torch.cat([x.image_color_similarity for x in gt_instances])
                    image_color_similarity = image_color_similarity[gt_inds].to(dtype=mask_feats.dtype)

                    loss_prj_term = compute_project_term(mask_scores, gt_bitmasks)

                    pairwise_losses = compute_pairwise_term(
                        mask_logits, self.pairwise_size,
                        self.pairwise_dilation
                    )

                    # image_similarity = mask_feat_similarity * image_color_similarity
                    image_similarity = image_color_similarity
                    weights = (image_similarity >= self.pairwise_color_thresh).float() * gt_bitmasks.float()
                    loss_pairwise = (pairwise_losses * weights).sum() / weights.sum().clamp(min=1.0)

                    warmup_factor = min(self._iter.item() / float(self._warmup_iters), 1.0)
                    loss_pairwise = loss_pairwise * warmup_factor
                    
                    # compute pseudo loss
                    proto_masks_list = []
                    self.prototypes.data.copy_(l2_normalize(self.prototypes))
                    mask_feats_ema = l2_normalize(mask_feats_ema).permute(0,2,3,1)

                    for i, x in enumerate(gt_instances):
                        masks = torch.einsum('hwd,npd->nphw', mask_feats_ema[i], self.prototypes[x.gt_classes]).detach()
                        proto_masks_list.append(masks)
                    proto_masks = torch.cat([x for x in proto_masks_list])
                    proto_masks = proto_masks[gt_inds]
                    

                    mask_scores_mean = scatter_mean(mask_scores_ema, gt_inds.unsqueeze(1), dim=0)[gt_inds]
                    # compute the threshold for pseudo labels
                    masks = (mask_scores > 0.5) * gt_bitmasks.float()
                    if self._iter > 20000:
                        n_pos = torch.sum(masks, dim=(2,3)).long() 
                        pseudo_seg = torch.amax(proto_masks, dim=1).unsqueeze(1).sigmoid()
                        pseudo_seg = 0.5 * pseudo_seg + 0.5 * mask_scores_mean
                        pseudo_scores = (pseudo_seg * gt_bitmasks).reshape(len(gt_inds), -1) 
                        pseudo_scores_rank = torch.sort(pseudo_scores, descending=True)[0]
                        thr = torch.gather(pseudo_scores_rank, dim=1, index=n_pos)[:,:,None,None]
                        # thr = scatter_mean(thr.squeeze(1), gt_inds)[gt_inds][:,None,None,None]
                        thr = torch.clamp(thr, min=0.5, max=0.70)
                        pseudo_seg_final = ( pseudo_seg > 0.5) * gt_bitmasks.float()
                    
                        # neg = (mask_scores < 0.5) 
                        # show_feature_map(pseudo_seg_final.detach(), 2)
                        # show_feature_map(mask_scores.detach(), 3)
                        # show_feature_map(neg.detach(), 4)
                        # pdb.set_trace()
                    
                        warmup_factor_2 = min(self._iter.item() / float(40000), 1.0)
                        weights = ((pseudo_seg > thr) | (pseudo_seg < 0.4)) * gt_bitmasks
                        loss_pseudo = (mask_focal_loss(mask_scores, pseudo_seg_final.detach(), weights)) * warmup_factor_2
                        losses["loss_pseudo"] = loss_pseudo

                    # update the prototypes
                    gt_classes = [x.gt_classes for x in gt_instances]
                    self.prototype_learning(mask_feats_ema, proto_masks, masks, gt_classes, pred_instances.labels, pred_instances.im_inds)
           
                    losses.update({
                        "loss_prj": loss_prj_term,
                        "loss_pairwise": loss_pairwise,
                    })
                else:
                    # fully-supervised CondInst losses
                    mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                    loss_mask = mask_losses.mean()
                    losses["loss_mask"] = loss_mask

            return losses
        else:
            if len(pred_instances) > 0:
                mask_logits = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_logits.sigmoid()

            return pred_instances
