_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=False)),
    roi_head=dict(
        bbox_head=dict(type='Shared4Conv1FCBBoxHead', conv_out_channels=256)))

# specify where the direct pre-trained model
load_from = 'work_dirs/mask_rcnn_r50_fpn_direct_pretrain_1x_coco/epoch_12.pth'
