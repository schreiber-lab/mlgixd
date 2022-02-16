# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from math import log

import torch
from torch import Tensor

_BBOX_XFORM_CLIP: float = log(1000. / 16)


def encode_boxes(reference_boxes: Tensor, proposals: Tensor) -> Tensor:
    """
    Encode a set of proposals with respect to some
    reference boxes
    Args:
        reference_boxes (Tensor): reference boxes
        proposals (Tensor): boxes to be encoded
    """

    x_ref = (reference_boxes[:, 0] + reference_boxes[:, 2]) / 2
    y_ref = (reference_boxes[:, 1] + reference_boxes[:, 3]) / 2
    w_ref = (reference_boxes[:, 2] - reference_boxes[:, 0])
    h_ref = (reference_boxes[:, 3] - reference_boxes[:, 1])

    x_pr = (proposals[:, 0] + proposals[:, 2]) / 2
    y_pr = (proposals[:, 1] + proposals[:, 3]) / 2
    w_pr = (proposals[:, 2] - proposals[:, 0])
    h_pr = (proposals[:, 3] - proposals[:, 1])

    targets_dx = (x_pr - x_ref) / w_ref
    targets_dy = (y_pr - y_ref) / h_ref
    targets_dw = torch.log(w_pr / w_ref)
    targets_dh = torch.log(h_pr / h_ref)

    targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)
    return targets


def decode_boxes(rel_codes: Tensor, boxes: Tensor):
    """
    From a set of original boxes and encoded relative box offsets,
    get the decoded boxes.
    Args:
        rel_codes (Tensor): encoded boxes
        boxes (Tensor): reference boxes.
    """
    device, dtype = rel_codes.device, rel_codes.dtype

    boxes = boxes.to(dtype)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = rel_codes[:, 0]
    dy = rel_codes[:, 1]
    dw = rel_codes[:, 2]
    dh = rel_codes[:, 3]

    # Prevent sending too large values into torch.exp()
    dw = torch.clamp(dw, max=_BBOX_XFORM_CLIP)
    dh = torch.clamp(dh, max=_BBOX_XFORM_CLIP)

    p_x = dx * widths + ctr_x
    p_y = dy * heights + ctr_y
    p_w = torch.exp(dw) * widths
    p_h = torch.exp(dh) * heights

    pred_boxes = torch.stack((p_x - p_w / 2, p_y - p_h / 2, p_x + p_w / 2, p_y + p_h / 2), dim=1)
    return pred_boxes


if __name__ == '__main__':
    num_boxes = torch.randint(0, 20, (10,))
    anchor_boxes = [torch.rand(n, 4) * 512 for n in num_boxes]
    close_boxes = [boxes + torch.rand(*boxes.shape) * 10 for boxes in anchor_boxes]

    anchor_boxes_cat = torch.cat(anchor_boxes, 0)
    close_boxes_cat = torch.cat(close_boxes, 0)

    indices = torch.where(
        (anchor_boxes_cat[:, 2] - anchor_boxes_cat[:, 0] > 0) &
        (anchor_boxes_cat[:, 3] - anchor_boxes_cat[:, 1] > 0) &
        (close_boxes_cat[:, 2] - close_boxes_cat[:, 0] > 0) &
        (close_boxes_cat[:, 3] - close_boxes_cat[:, 1] > 0)
    )

    anchor_boxes_cat, close_boxes_cat = anchor_boxes_cat[indices], close_boxes_cat[indices]

    encoded_boxes = encode_boxes(anchor_boxes_cat, close_boxes_cat)
    decoded_boxes = decode_boxes(encoded_boxes, anchor_boxes_cat)

    assert torch.allclose(decoded_boxes, close_boxes_cat)
