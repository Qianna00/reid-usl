import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS, build_loss


@HEADS.register_module()
class StandardBaselineHead(nn.Module):
    """The head outputs re-ID features `f` and ID prediction logits `p`.
    Re-ID features `f` is used to calculate triplet loss. ID prediction
    logits `p` is used to calculate cross entropy loss. The margin `m` of
    triplet loss is set to be 0.3.
    """

    def __init__(self,
                 num_classes,
                 feat_dim=2048,
                 cls_loss=dict(type='CrossEntropyLoss'),
                 triplet_loss=dict(type='TripletLoss', m=0.3)):
        super(StandardBaselineHead, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.classifier = nn.Linear(self.feat_dim, self.num_classes)

        # build losses
        self.cls_loss = build_loss(cls_loss)
        self.triplet_loss = build_loss(triplet_loss)

    def init_weights(self, **kwargs):
        normal_init(self.classifier, std=0.01)

    def forward(self, feats, labels):
        """Forward function.
        """
        logits = self.classifier(feats)
        loss_cls = self.cls_loss(logits, labels)

        loss_triplet = self.triplet_loss(feats, labels)

        return dict(loss_cls=loss_cls, loss_triplet=loss_triplet)


@HEADS.register_module()
class StrongBaselineHead(StandardBaselineHead):

    def forward(self, f_t, f_i, labels):
        """Forward function.

        Args:
            f_t (Tensor): Features before the BN layer.
            f_i (Tensor): Features after passing through the BN layer.
            labels (Tensor): Ground truths of samples.
        """
        logits = self.classifier(f_i)
        loss_cls = self.cls_loss(logits, labels)

        loss_triplet = self.triplet_loss(f_t, labels)

        return dict(loss_cls=loss_cls, loss_triplet=loss_triplet)
