_base_ = [
    '../_base_/models/bot.py', '../_base_/evaluation.py',
    '../_base_/default_runtime.py'
]
data_source_cfg = dict(
    type='Market1501', data_root='/data/datasets/market1501')
dataset_type = 'ReIDDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='RandomCrop', size=(256, 128), padding=10),
    dict(type='RandomHorizontalFlip'),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
test_pipeline = [
    dict(type='Resize', size=(256, 128), interpolation=3),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg)
]
data = dict(
    samples_per_gpu=64,
    workers_per_gpu=4,
    sampler=dict(type='IdentitySampler', num_instances=4),
    train=dict(
        type=dataset_type,
        data_source=data_source_cfg,
        pipeline=train_pipeline),
    test=dict(
        type=dataset_type,
        data_source=data_source_cfg,
        pipeline=test_pipeline,
        test_mode=True))

# optimizer
optimizer = dict(type='Adam', lr=0.00035, weight_decay=0.0005)
# learning policy
lr_config = dict(policy='step', step=[40, 70])
total_epochs = 120
