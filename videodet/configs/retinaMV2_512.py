# model settings
model = dict(
    type='SingleStageDetector',
    pretrained='zoo/mobilenet_v2.pth.tar',
    backbone=dict(
        type='SSDMobileNetV2',
        input_size=-1,
        frozen_stages=3,
        out_layers=('layer4', 'layer7', 'layer14', 'layer19'),
        with_extra=False,
        norm_eval=True,
        ),
    neck=dict(
        type='FPN',
        in_channels=[24, 32, 96, 1280],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=31,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=0.11, loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_thr=0.5),
    max_per_img=100)
# dataset settings
vid_dataset_type = 'StillVIDDataset'
det_dataset_type = 'DET30Dataset'
data_root = 'data/ILSVRC2015/'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=2,
    train=[
        dict(
            type=vid_dataset_type,
            ann_file=data_root + 'ImageSets/VID/VID_train_15frames.txt',
            img_prefix=data_root,
            pipeline=train_pipeline),
        dict(
            type=det_dataset_type,
            ann_file=data_root + 'ImageSets/VID/DET_train_30classes.txt',
            img_prefix=data_root,
            pipeline=train_pipeline),
    ],
    val=dict(
        type=vid_dataset_type,
        ann_file=data_root + 'ImageSets/VID/VID_val_frames.txt',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=vid_dataset_type,
        ann_file=data_root + 'ImageSets/VID/VID_val_frames.txt',
        img_prefix=data_root,
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
evaluation = dict(interval=12, num_evals=5000*4, shuffle=True)
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = './workvids/retinaMV2_512'
load_from = None
resume_from = None
workflow = [('train', 1)]
