import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import get_dist_info

from reid.core import cosine_dist, euclidean_dist
from reid.utils import concat_all_gather
from ..builder import LOSSES
from ..utils import GatherLayer

MAX_INF = 99999999


def triplet_loss(dist, is_pos, is_neg, margin):
    """
    Args:
        dist (Tensor): Distances between samples.
        labels (Tensor): Labels of the samples.
    """
    dist_ap, _ = torch.max(dist * is_pos, dim=1)
    dist_an, _ = torch.min(dist * is_neg + is_pos * MAX_INF, dim=1)

    y = torch.ones_like(dist_an)
    loss = F.margin_ranking_loss(dist_an, dist_ap, y, margin=margin)

    return loss


@LOSSES.register_module()
class TripletLoss(nn.Module):
    """Triplet loss.

    Args:
        metric (str): Which metric is used to measure distances.
        feat_norm (bool, optional): Whether to normalize features.
            Default: False.
        m (float, optional): Margin value in the triplet loss. Default: 0.3.
        loss_weight (float, optional): Weight of the loss. Default: 1.0.
    """

    def __init__(self,
                 metric='euclidean',
                 feat_norm=False,
                 m=0.3,
                 loss_weight=1.0):
        super(TripletLoss, self).__init__()

        assert metric in ('cosine', 'euclidean'), \
            f'{metric} distance is unsupported'
        if metric == 'cosine':
            assert feat_norm, 'feat_norm is required for cosine distance'
        self.metric = cosine_dist if metric == 'cosine' else euclidean_dist
        self.feat_norm = feat_norm

        self.m = m
        self.loss_weight = loss_weight

    def forward(self, feats, labels):
        """Forward function.

        Args:
            feats (Tensor): Inputed features for computing loss.
            labels (Tensor): Labels of the samples.
        """
        rank, world_size = get_dist_info()
        if world_size > 1:
            feats = torch.cat(GatherLayer.apply(feats), dim=0)
            labels = concat_all_gather(labels)

        # compute distance
        if self.feat_norm:
            feats = F.normalize(feats, dim=1)
        dist = self.metric(feats, feats)

        # labels
        N = dist.shape[0]
        labels = labels.unsqueeze(1).expand(N, N)
        is_pos = labels.eq(labels.t()).float()
        is_neg = labels.ne(labels.t()).float()

        if world_size > 1:
            dist = dist.view(world_size, -1, N)[rank]
            is_pos = is_pos.view(world_size, -1, N)[rank]
            is_neg = is_neg.view(world_size, -1, N)[rank]

        loss_triplet = triplet_loss(dist, is_pos, is_neg, margin=self.m)

        return loss_triplet
