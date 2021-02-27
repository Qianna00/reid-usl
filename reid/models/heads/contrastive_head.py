import torch
import torch.nn as nn

from ..builder import HEADS


@HEADS.register_module()
class ContrastiveHead(nn.Module):

    def __init__(self, temperature=0.1):
        super(ContrastiveHead, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pos, neg):
        """Forward function.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): NxK negative similarity.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.shape[0]
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature

        labels = torch.zeros((N, ), dtype=torch.long).cuda()
        loss = self.criterion(logits, labels)

        return dict(loss=loss)
