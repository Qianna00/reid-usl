_base_ = [
    '../_base_/models/bot.py', '../_base_/evaluation.py',
    '../_base_/default_runtime.py'
]

model = dict(
    type='MMCL',
    feat_dim=2048,
    memory_size=12936,
    base_momentum=0.5,
    t=0.6,
    start_epoch=5,
    head=dict(type='MMCLHead', delta=5.0, r=0.01))

data_source_cfg = dict(
    type='Market1501', data_root='/data/datasets/market1501')
dataset_type = 'ReIDDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='RandomCamStyle',
        camstyle_root='bounding_box_train_camstyle',
        p=0.3),
    dict(
        type='RandomResizedCropV2',
        size=(256, 128),
        scale=(0.64, 1.0),
        ratio=(2., 3.),
        interpolation=3),
    dict(type='RandomHorizontalFlip'),
    dict(type='RandomRotation', degrees=10),
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    # +REA
    dict(
        type='RandomErasingV2',
        p=0.5,
        scale=(0.02, 0.4),
        ratio=(0.3, 3.33),
        value=(0.485, 0.456, 0.406))
]
test_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
data = dict(
    samples_per_gpu=32,  # 32 x 4 = 128
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=data_source_cfg,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_source=data_source_cfg,
        pipeline=test_pipeline,
        test_mode=True))
evaluation = dict(metric='euclidean', feat_norm=False)

# additional hooks
custom_hooks = [dict(type='MMCLHook')]
# optimizer
paramwise_cfg = dict(custom_keys={'backbone': dict(lr_mult=0.1)})
optimizer = dict(
    type='SGD',
    lr=0.1,
    weight_decay=0.0005,
    momentum=0.9,
    paramwise_cfg=paramwise_cfg)
# learning policy
lr_config = dict(policy='step', step=[40])
total_epochs = 60
