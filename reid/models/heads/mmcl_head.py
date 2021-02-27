import torch
import torch.nn as nn

from ..builder import HEADS


@HEADS.register_module()
class MMCLHead(nn.Module):
    """MMCL head.

    Args:
        delta (float): Coefficient for MMCL. Default: 5.0.
        r (float): Hard negative mining ratio. Default: 0.01.
    """

    def __init__(self, delta=5.0, r=0.01, **kwargs):
        super(MMCLHead, self).__init__()
        self.delta = delta
        self.r = r

    def init_weights(self, **kwargs):
        pass

    def forward(self, logits, labels):
        """Forward function.

        Args:
            logits (Tensor): Similarity between samples.
            labels (Tensor): Memory-based multi-labels.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = logits.shape[0]
        loss = []

        for i in range(N):
            logit = logits[i]
            label = labels[i]

            pos_logit = torch.masked_select(logit, label == 1)
            neg_logit = torch.masked_select(logit, label == 0)

            _, idx = torch.sort(neg_logit.clone().detach(), descending=True)
            num = int(self.r * neg_logit.shape[0])
            mask = neg_logit.new_zeros(neg_logit.size(), dtype=torch.bool)
            mask[idx[:num]] = 1
            hard_neg_logit = torch.masked_select(neg_logit, mask == 1)

            _loss = self.delta * torch.mean((1. - pos_logit).pow(2)) + \
                torch.mean((1 + hard_neg_logit).pow(2))
            loss.append(_loss)

        loss = sum(loss) / N

        return dict(loss=loss)
