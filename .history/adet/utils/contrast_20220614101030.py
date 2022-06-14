import torch
import torch.nn as nn
import torch.nn.functional as F


def momentum_update(old_value, new_value, momentum, debug=False):
    update = momentum * old_value + (1 - momentum) * new_value
    if debug:
        print("old prot: {:.3f} x |{:.3f}|, new val: {:.3f} x |{:.3f}|, result= |{:.3f}|".format(
            momentum, torch.norm(old_value, p=2), (1 - momentum), torch.norm(new_value, p=2),
            torch.norm(update, p=2)))
    return update


def l2_normalize(x):
    red_global_masks = F.interpolate(
                x,
                scale_factor=2,
                mode="bilinear", align_corners=False)
    return F.normalize(x, p=2, dim=1)