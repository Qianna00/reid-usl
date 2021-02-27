model = dict(
    type='Baseline',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[3],
        strides=(1, 2, 2, 2),  # last stride: 2
        norm_cfg=dict(type='BN'),
        norm_eval=False),
    neck=dict(type='AvgPoolNeck'),
    head=dict(
        type='StandardBaselineHead',
        num_classes=751,  # for Market1501
        cls_loss=dict(type='CrossEntropyLoss'),
        triplet_loss=dict(type='TripletLoss')))
