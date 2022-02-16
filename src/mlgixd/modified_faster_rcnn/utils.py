# -*- coding: utf-8 -*-
#
#
# This source code is licensed under the GPL license found in the
# LICENSE file in the root directory of this source tree.

from typing import (
    List,
    Tuple,
    Union,
    Callable,
)

import torch
from torch import Tensor


class BalancedPositiveNegativeSampler(object):
    """
    This class samples batches, ensuring that they contain a fixed proportion of positives
    """

    def __init__(self, batch_size_per_image: int, positive_fraction: float):
        """
        Args:
            batch_size_per_image (int): number of elements to be selected per image
            positive_fraction (float): percentace of positive elements per batch
        """
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction

    def __call__(self, matched_idxs: Union[List[Tensor], Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Args:
            matched idxs: list of tensors containing -1, 0 or positive values.
                Each tensor corresponds to a specific image.
                -1 values are ignored, 0 are considered as negatives and > 0 as
                positives.

        Returns:
            pos_idx (list[tensor])
            neg_idx (list[tensor])

        Returns two lists of binary masks for each image.
        The first list contains the positive elements that were selected,
        and the second list the negative example.

        """
        pos_idx = []
        neg_idx = []
        for matched_idxs_per_image in matched_idxs:
            positive = torch.where(matched_idxs_per_image >= 1)[0]
            negative = torch.where(matched_idxs_per_image == 0)[0]

            num_pos = int(self.batch_size_per_image * self.positive_fraction)
            # protect against not enough positive examples
            num_pos = min(positive.numel(), num_pos)
            num_neg = self.batch_size_per_image - num_pos
            # protect against not enough negative examples
            num_neg = min(negative.numel(), num_neg)

            # randomly select positive and negative examples
            perm1 = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
            perm2 = torch.randperm(negative.numel(), device=negative.device)[:num_neg]

            pos_idx_per_image = positive[perm1]
            neg_idx_per_image = negative[perm2]

            # create binary mask from indices
            pos_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )
            neg_idx_per_image_mask = torch.zeros_like(
                matched_idxs_per_image, dtype=torch.bool
            )

            pos_idx_per_image_mask[pos_idx_per_image] = True
            neg_idx_per_image_mask[neg_idx_per_image] = True

            pos_idx.append(pos_idx_per_image_mask)
            neg_idx.append(neg_idx_per_image_mask)

        sampled_pos_inds = torch.cat(pos_idx, dim=0)
        sampled_neg_inds = torch.cat(neg_idx, dim=0)

        sampled_inds = (sampled_pos_inds + sampled_neg_inds).type(torch.bool)

        return sampled_inds, sampled_pos_inds


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the MxN match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns a tensor of size N containing the index of the ground-truth
    element m that matches to prediction n. If there is no match, a negative value
    is returned.
    """

    BELOW_LOW_THRESHOLD = -1
    BETWEEN_THRESHOLDS = -2

    __annotations__ = {
        'BELOW_LOW_THRESHOLD': int,
        'BETWEEN_THRESHOLDS': int,
    }

    def __init__(self, high_threshold: float, low_threshold: float, allow_low_quality_matches: bool = False):
        """
        Args:
            high_threshold (float): quality values greater than or equal to
                this value are candidate matches.
            low_threshold (float): a lower quality threshold used to stratify
                matches into three levels:
                1) matches >= high_threshold
                2) BETWEEN_THRESHOLDS matches in [low_threshold, high_threshold)
                3) BELOW_LOW_THRESHOLD matches in [0, low_threshold)
            allow_low_quality_matches (bool): if True, produce additional matches
                for predictions that have only low-quality match candidates. See
                set_low_quality_matches_ for more details.
        """
        self.BELOW_LOW_THRESHOLD = -1
        self.BETWEEN_THRESHOLDS = -2
        assert low_threshold <= high_threshold
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix: Tensor) -> Tensor:
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
            pairwise quality between M ground-truth elements and N predicted elements.

        Returns:
            matches (Tensor[int64]): an N tensor where N[i] is a matched gt in
            [0, M - 1] or a negative value indicating that prediction i could not
            be matched.
        """
        if match_quality_matrix.numel() == 0:
            # empty targets or proposals not supported during training
            if match_quality_matrix.shape[0] == 0:
                raise ValueError(
                    "No ground-truth boxes available for one of the images "
                    "during training")
            else:
                raise ValueError(
                    "No proposal boxes available for one of the images "
                    "during training")

        # match_quality_matrix is M (gt) x N (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)
        if self.allow_low_quality_matches:
            all_matches = matches.clone()
        else:
            all_matches = None

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self.low_threshold
        between_thresholds = (matched_vals >= self.low_threshold) & (
                matched_vals < self.high_threshold
        )
        matches[below_low_threshold] = self.BELOW_LOW_THRESHOLD
        matches[between_thresholds] = self.BETWEEN_THRESHOLDS

        if self.allow_low_quality_matches:
            assert all_matches is not None
            self.set_low_quality_matches_(matches, all_matches, match_quality_matrix)

        return matches

    def set_low_quality_matches_(self, matches, all_matches, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth with which it has the highest
        quality value.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.where(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[1]
        matches[pred_inds_to_update] = all_matches[pred_inds_to_update]


def assign_targets_to_anchors(
        proposal_matcher: Matcher,
        anchors: List[Tensor],
        targets: List[Tensor],
        box_similarity: Callable[[Tensor, Tensor], Tensor]
) -> Tuple[List[Tensor], List[Tensor]]:
    labels = []
    matched_gt_boxes = []

    for anchors_per_image, gt_boxes_per_image in zip(anchors, targets):
        if gt_boxes_per_image.numel() == 0:
            # Background image (negative example)
            device = anchors_per_image.device
            matched_gt_boxes_per_image = torch.zeros(anchors_per_image.shape, dtype=torch.float32, device=device)
            labels_per_image = torch.zeros((anchors_per_image.shape[0],), dtype=torch.float32, device=device)
        else:
            match_quality_matrix = box_similarity(gt_boxes_per_image, anchors_per_image)
            matched_idxs = proposal_matcher(match_quality_matrix)
            # get the targets corresponding GT for each proposal
            # NB: need to clamp the indices because we can have a single
            # GT in the image, and matched_idxs can be -2, which goes
            # out of bounds
            matched_gt_boxes_per_image = gt_boxes_per_image[matched_idxs.clamp(min=0)]

            labels_per_image = matched_idxs >= 0
            labels_per_image = labels_per_image.to(dtype=torch.float32)

            # Background (negative examples)
            bg_indices = matched_idxs == proposal_matcher.BELOW_LOW_THRESHOLD
            labels_per_image[bg_indices] = 0.0

            # discard indices that are between thresholds
            inds_to_discard = matched_idxs == proposal_matcher.BETWEEN_THRESHOLDS
            labels_per_image[inds_to_discard] = -1.0

        labels.append(labels_per_image)
        matched_gt_boxes.append(matched_gt_boxes_per_image)
    return labels, matched_gt_boxes


def get_top_n_idx(k: int, score: Tensor, num_anchors_per_level: List[int]) -> Tensor:
    """
    Get top k indices with respect to score independently for each image and each feature map.

    Args:
        k (int): Number of top scores per feature map.
        score (Tensor): Score tensor of shape (img_num, sum(num_anchors_per_level)).
        num_anchors_per_level (List[int]): Number of anchors per feature map.

    Returns:
        Tensor: indices of shape (img_num, k * len(num_anchors_per_level))

    >>> k_ = 2
    >>> num_anchors_per_level_ = [6, 4]
    >>> score_ = torch.stack([torch.arange(10), 10 - torch.arange(10)])
    >>> num_images = score_.shape[0]
    >>> score_
    tensor([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9],
            [10,  9,  8,  7,  6,  5,  4,  3,  2,  1]])

    >>> indices = get_top_n_idx(k_, score_, num_anchors_per_level_)
    >>> indices
    tensor([[5, 4, 9, 8],
            [0, 1, 6, 7]])

    >>> image_range = torch.arange(num_images)
    >>> batch_idx = image_range[:, None]
    >>> score_[batch_idx, indices]
    tensor([[ 5,  4,  9,  8],
            [10,  9,  4,  3]])
    """

    r = []
    offset = 0

    for ob in score.split(num_anchors_per_level, 1):
        num_anchors = ob.shape[1]
        k = min(k, num_anchors)
        _, top_n_idx = ob.topk(k, dim=1)
        r.append(top_n_idx + offset)
        offset += num_anchors

    return torch.cat(r, dim=1)


def get_levels(img_num: int, num_anchors_per_level: List[int], device: torch.device) -> Tensor:
    """
    Returns a 2D tensor of shape (img_num, sum(num_anchors_per_level)) with indices corresponding
    to source feature maps. If img_num = 0 or None, returns a 1D tensor of shape (sum(num_anchors_per_level), ).


    Args:
        img_num (int): Number of images.
        num_anchors_per_level (List[int]): Number of anchors per feature map.
        device (torch.device): Device to allocate the result.

    Returns:
        Tensor: Feature map indices.

    """
    levels = [
        torch.full((n,), idx, dtype=torch.int64, device=device)
        for idx, n in enumerate(num_anchors_per_level)
    ]
    levels = torch.cat(levels, 0)
    if img_num:
        levels = levels.reshape(1, -1).repeat(img_num, 1)
    return levels


def num_per_level_from_levels(levels: Tensor, num_of_levels: int) -> Tensor:
    level_numbers = torch.arange(num_of_levels, device=levels.device)
    indices = torch.cat([torch.searchsorted(levels, level_numbers),
                         torch.tensor(levels.size(), device=levels.device)])
    return indices[1:] - indices[:-1]


def valid_boxes(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])


if __name__ == '__main__':
    import doctest

    doctest.testmod()
