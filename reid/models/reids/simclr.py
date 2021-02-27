import torch
import torch.nn.functional as F

from ..base import BaseModel
from ..builder import REIDS, build_backbone, build_head, build_neck
from ..utils import GatherLayer


@REIDS.register_module()
class SimCLR(BaseModel):
    """SimCLR.

    Implementation of ``A Simple Framework for Contrastive Learning
    of Visual Representations <https://arxiv.org/abs/2002.05709>``.

    Args:
        backbone (dict): Config dict of backbone.
        neck (dict): Config dict of neck.
        head (dict): Config dict of head.
        pretrained (str, optional): Path to pre-trained weights.
            Default: None.
    """

    def __init__(self, backbone, neck=None, head=None, pretrained=None):
        super(SimCLR, self).__init__()
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.head = build_head(head)

        self.init_weights(pretrained=pretrained)

    @staticmethod
    def _create_buffer(N):
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).cuda()
        pos_ind = (torch.arange(N * 2, dtype=torch.long).cuda(),
                   2 * torch.arange(N, dtype=torch.long).unsqueeze(1).repeat(
                       1, 2).view(-1, 1).squeeze().cuda())
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).cuda()
        neg_mask[pos_ind] = 0

        return mask, pos_ind, neg_mask

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Default: None.
        """
        super(SimCLR, self).init_weights(pretrained=pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.neck.init_weights(init_linear='kaiming')

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape
                (N, 2, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, f'img should have 5 dims, but got: {img.dim()}'
        img = img.reshape(img.shape[0] * 2, img.shape[2], img.shape[3],
                          img.shape[4])
        x = self.backbone(img)
        z = self.neck(x)[0]
        z = F.normalize(z, dim=1)

        z = torch.cat(GatherLayer.apply(z), dim=0)  # 2N
        assert z.shape[0] % 2 == 0
        N = z.shape[0] // 2

        s = torch.matmul(z, z.permute(1, 0))  # 2N x 2N
        mask, pos_ind, neg_mask = self._create_buffer(N)
        # remove diagonal, 2N x (2N - 1)
        s = torch.masked_select(s, mask == 1).reshape(s.shape[0], -1)
        positive = s[pos_ind].unsqueeze(1)  # 2N x 1
        # select negative, 2N x (2N - 2)
        negative = torch.masked_select(s,
                                       neg_mask == 1).reshpae(s.shape[0], -1)
        losses = self.head(positive, negative)

        return losses

    def forward_test(self, img, **kwargs):
        """Extract features for testing.
        """
        x = self.backbone(img)
        z = self.neck(x)[0]

        return z
