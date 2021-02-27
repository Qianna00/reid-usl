import torch
import torch.nn as nn

from ..base import BaseModel
from ..builder import REIDS, build_backbone, build_head, build_neck


@REIDS.register_module()
class BYOL(BaseModel):
    """BYOL.

    Implementation of ``Bootstrap your own latent: A new approach to
    self-supervised learning <https://arxiv.org/abs/2006.07733>``.

    Args:
        backbone (dict): Config dict of backbone.
        neck (dict): Config dict of neck.
        head (dict): Config dict of head.
        pretrained (str, optional): Path to pre-trained weights.
            Default: None.
        base_momentum (float, optional): The base momentum coefficient for
            the target network. Default: 0.996.
    """

    def __init__(self,
                 backbone,
                 neck,
                 head,
                 pretrained=None,
                 base_momentum=0.996):
        super(BYOL, self).__init__()
        self.online_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.target_net = nn.Sequential(
            build_backbone(backbone), build_neck(neck))
        self.head = build_head(head)

        for param in self.target_net.parameters():
            param.requires_grad = False

        self.init_weights(pretrained=pretrained)

        self.base_momentum = base_momentum
        self.momentum = base_momentum

    def init_weights(self, pretrained=None):
        """Initialize the weights of model.
        """
        super().init_weights(pretrained=pretrained)
        self.online_net[0].init_weights(pretrained=pretrained)
        self.online_net[1].init_weights(init_linear='kaiming')
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data.copy_(param_ol.data)
        # head
        self.head.init_weights()

    @torch.no_grad()
    def momentum_update(self):
        """Momentum update of the target network.
        """
        for param_ol, param_tgt in zip(self.online_net.parameters(),
                                       self.target_net.parameters()):
            param_tgt.data = param_tgt.data * self.momentum + \
                param_ol.data * (1. - self.momentum)

    def forward_train(self, img, **kwargs):
        """Forward computation during training.

        Args:
            img (Tensor): Input of two concatenated images of shape
                (N, 2, C, H, W). Typically these should be mean centered
                and std scaled.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert img.dim() == 5, f'Input must be 5 dims, but got: {img.dim()}'
        im_v1 = img[:, 0, ...].contiguous()
        im_v2 = img[:, 1, ...].contiguous()

        # compute query
        proj_online_v1 = self.online_net(im_v1)[0]
        proj_online_v2 = self.online_net(im_v2)[0]
        with torch.no_grad():
            proj_target_v1 = self.target_net(im_v1)[0].clone().detach()
            proj_target_v2 = self.target_net(im_v2)[0].clone().detach()

        loss_v1v2 = self.head(proj_online_v1, proj_target_v2)
        loss_v2v1 = self.head(proj_online_v2, proj_target_v1)

        losses = {}
        for key in loss_v1v2.keys():
            losses[key] = loss_v1v2[key] + loss_v2v1[key]

        return losses

    def forward_test(self, img, **kwargs):
        return self.online_net(img)[0]
