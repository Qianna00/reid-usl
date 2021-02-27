model = dict(
    type='BoT',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        strides=(1, 2, 2, 1),  # last stride: 1
        norm_cfg=dict(type='BN'),
        norm_eval=False),
    neck=dict(
        type='BNNeck',
        feat_dim=2048,
        norm_cfg=dict(type='BN1d'),
        with_avg_pool=True,
        avgpool=dict(type='AvgPoolNeck')),
    head=dict(
        type='StrongBaselineHead',
        num_classes=751,  # for Market1501
        cls_loss=dict(type='CrossEntropyLoss'),
        triplet_loss=dict(type='TripletLoss')))
