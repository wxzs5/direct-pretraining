_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    pretrained=None,
    backbone=dict(norm_cfg=dict(type='BN', requires_grad=False)))

# specify where the direct pre-trained model
load_from = 'work_dirs/cascade_rcnn_r50_fpn_direct_pretrain_1x_coco/epoch_12.pth'
