_base_ = './coco_detection.py'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(448, 448),
        ratio_range=(0.8, 1.2),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(448, 448)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(448, 448)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=4,
        dataset=dict(
            type='CocoDataset',
            ann_file='data/coco/annotations/instances_train2017.json',
            img_prefix='data/coco/train2017/',
            pipeline=train_pipeline)))
