from mmcv.utils import Registry, build_from_cfg

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
LOSSES = Registry('loss')
REIDS = Registry('reid')


def build(cfg, registry):
    return build_from_cfg(cfg, registry)


def build_backbone(cfg):
    return build(cfg, BACKBONES)


def build_neck(cfg):
    return build(cfg, NECKS)


def build_head(cfg):
    return build(cfg, HEADS)


def build_loss(cfg):
    return build(cfg, LOSSES)


def build_reid(cfg):
    return build(cfg, REIDS)
