import os

model = dict(
    type='RetinaFace',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[512, 1024, 2048],
        out_channels=256,
        num_outs=3),
    bbox_head=dict(
        type='RetinaFaceHead',
        in_channels=256,
        feat_channels=256,
        anchor_ratios=[1.0],
        anchor_scales=[2, 16],
        anchor_strides=[8, 16, 32],
        loss_bbox=dict(
            type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0)),
)
# training and testing settings
cudnn_benchmark = True
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.35,
        neg_iou_thr=0.35,
        min_pos_iou=0.,
        ignore_iof_thr=-1,
        gt_max_assign_all=False),
    sampler=dict(
        type='RandomSampler',
        num=256,
        pos_fraction=0.5,
        neg_pos_ub=-1,
        add_gt_as_proposals=False),
    target_stds=(0.1, 0.1, 0.2, 0.2),
    allowed_border=0,
    pos_weight=-1,
)
test_cfg = dict()
dataset_type = 'WIDERRetinaFaceDataset'
data_root = '/data/Datasets/widerface'
train_pipeline = [
    dict(
        type='WiderFacePreProc',
        img_size=864,
        means=(104, 117, 123),
        std=(1.0, 1.0, 1.0),
        to_rgb=False),
    dict(type='WiderRetinaFaceFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'boxes', 'landms', 'labels'],
        meta_keys=['img_norm_cfg', 'img_shape', 'pad_shape']),
]
test_pipeline = [
    dict(type='ImageToTensor', keys=['img']),
]
data = dict(
    imgs_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        txt_path=os.path.join(data_root, 'train/label.txt'),
        ann_file=None,
        pipeline=train_pipeline
        ),
    val=dict(
        type=dataset_type,
        txt_path=os.path.join(data_root, 'val/label.txt'),
        ),
    test=dict(
        type=dataset_type,
        txt_path=os.path.join(data_root, 'test/label.txt'),
        ann_file=None,
        ))
# optimizer
optimizer = dict(type='SGD', lr=1e-3, momentum=0.9, weight_decay=5e-4)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1e-6,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 100
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './experiments'
load_from = None
resume_from = None
workflow = [('train', 1)]
