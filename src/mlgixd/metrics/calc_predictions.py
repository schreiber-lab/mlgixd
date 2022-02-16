# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from tqdm import trange

import torch
from torch import Tensor

from ..ml import batch_filter


@torch.no_grad()
def get_sim_predictions(
        model, dset,
        num: int = 1000,
        batch_num: int = 10,
        min_score: float = 0.1,
        max_nms: float = 0.1,
        disable_tqdm: bool = True,
):
    model.eval()

    predictions_list = []
    scores_list = []
    targets_list = []

    for _ in trange(num, disable=disable_tqdm):
        imgs, boxes = dset.get_batch(batch_num)

        predictions, scores = model(imgs)
        predictions, scores = batch_filter(predictions, scores, max_nms, min_score)

        targets_list += _list_to_cpu(boxes)
        scores_list += _list_to_cpu(scores)
        predictions_list += _list_to_cpu(predictions)

    return targets_list, predictions_list, scores_list


def _list_to_cpu(arr: List[Tensor]) -> List[Tensor]:
    return [a.detach().cpu() for a in arr]
