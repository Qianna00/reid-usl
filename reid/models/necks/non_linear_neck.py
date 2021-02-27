import torch.nn as nn
from mmcv.cnn import is_norm, kaiming_init, normal_init

from ..builder import NECKS, build_neck


@NECKS.register_module()
class NonLinearNeckV1(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    """

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 avgpool=dict(type='AvgPoolNeck')):
        super(NonLinearNeckV1, self).__init__()
        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avgpool = build_neck(avgpool)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear=init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x[0])]


@NECKS.register_module()
class NonLinearNeckV2(nn.Module):

    def __init__(self,
                 in_channels,
                 hid_channels,
                 out_channels,
                 with_avg_pool=True,
                 avgpool=dict(type='AvgPoolNeck')):
        super(NonLinearNeckV2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avgpool = build_neck(avgpool)

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels), nn.BatchNorm1d(hid_channels),
            nn.ReLU(inplace=True), nn.Linear(hid_channels, out_channels))

    def init_weights(self, init_linear='normal'):
        _init_weights(self, init_linear=init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)

        return [self.mlp(x[0])]


def _init_weights(module, init_linear='normal', std=0.01, bias=0):
    assert init_linear in ['normal', 'kaiming']

    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == 'normal':
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode='fan_in', nonlinearity='relu')
        elif is_norm(m):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
