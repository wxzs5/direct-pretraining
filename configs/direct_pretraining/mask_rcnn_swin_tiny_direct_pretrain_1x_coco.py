_base_ = [
    '../_base_/models/mask_rcnn_swin_fpn.py',
    '../_base_/datasets/coco_instance_direct_size448.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False),
    neck=dict(in_channels=[96, 192, 384, 768]),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256)))

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

# do not use mmdet version fp16
fp16 = None
optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)

lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    _delete_=True)
runner = dict(type='EpochBasedRunnerAmp')
