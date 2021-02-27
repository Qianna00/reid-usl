import torch.nn as nn
import torch.nn.functional as F

from ..builder import HEADS, build_neck


@HEADS.register_module()
class LatentPredictHead(nn.Module):

    def __init__(self, predictor, **kwargs):
        super(LatentPredictHead, self).__init__()
        self.predictor = build_neck(predictor)

    def init_weights(self, init_linear='normal'):
        self.predictor.init_weights(init_linear=init_linear)

    def forward(self, inputs, targets):
        """Forward function.

        Args:
            inputs (Tensor): NxC input features.
            targets (Tensor): NxC input features.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = inputs.shape[0]

        pred = self.predictor([inputs])[0]
        pred_norm = F.normalize(pred, dim=1)
        tgt_norm = F.normalize(targets, dim=1)

        loss = -2 * (pred_norm * tgt_norm).sum()
        loss /= N

        return dict(loss=loss)
