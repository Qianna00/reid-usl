import torch.nn as nn
from mmcv.cnn import build_norm_layer

from ..builder import NECKS, build_neck


@NECKS.register_module()
class BNNeck(nn.Module):
    """BN neck."""

    def __init__(self,
                 feat_dim=2048,
                 norm_cfg=dict(type='BN1d'),
                 with_avg_pool=True,
                 avgpool=dict(type='AvgPoolNeck')):
        super(BNNeck, self).__init__()
        # avgpool
        self.with_avg_pool = with_avg_pool
        if self.with_avg_pool:
            self.avgpool = build_neck(avgpool)

        # BN
        self.feat_dim = feat_dim
        self.norm_cfg = norm_cfg
        _, self.bn = build_norm_layer(norm_cfg, feat_dim)

    def init_weights(self, **kwargs):
        nn.init.constant_(self.bn.weight, 1.)
        nn.init.constant_(self.bn.bias, 0.)

    def forward(self, x, loc='after'):
        assert isinstance(x, (list, tuple)) and len(x) == 1
        assert loc in ('after', 'both')

        if self.with_avg_pool:
            # neck returns a list
            x = self.avgpool(x)[0]
            x = x.flatten(start_dim=1)

        if loc == 'after':
            return [self.bn(x)]
        elif loc == 'both':
            return [x, self.bn(x)]
        else:
            raise ValueError(f'unsupported loc: {loc}')
