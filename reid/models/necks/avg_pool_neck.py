import torch
import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class AvgPoolNeck(nn.Module):

    def __init__(self):
        super(AvgPoolNeck, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self, **kwargs):
        pass

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 1

        x = self.avgpool(x[0])
        x = x.flatten(start_dim=1)

        return [x]


@NECKS.register_module()
class GEMPoolNeck(AvgPoolNeck):

    def __init__(self, p=3., p_train=False, eps=1e-6):
        super(GEMPoolNeck, self).__init__()
        self.p_train = p_train
        self.p = nn.Parameter(torch.ones(1) * p) if p_train else p
        self.eps = eps

    def forward(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 1

        x = x[0].clamp(self.eps).pow(self.p)
        x = self.avgpool(x)
        x = x.pow(1. / self.p)
        x = x.flatten(start_dim=1)

        return [x]
