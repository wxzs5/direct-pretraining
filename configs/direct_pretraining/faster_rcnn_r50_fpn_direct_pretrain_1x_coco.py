_base_ = [
    '../_base_/datasets/coco_detection_direct_size448.py',
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(frozen_stages=-1, zero_init_residual=False, norm_eval=False),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256)))

# optimizer
optimizer = dict(lr=0.08, paramwise_cfg=dict(norm_decay_mult=0))
optimizer_config = dict(_delete_=True, grad_clip=None)
# learning policy
lr_config = dict(
    policy='fixed',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.01,
    _delete_=True)
