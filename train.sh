#!/bin/bash

# ps -ef|grep "python" |grep -v grep|cut -c 9-16|xargs sudo kill -9


# ./dist_train.sh experiments/mask_rcnn_swin_tiny_fine_dst_640p3x_1x_coco.py 8 --resume-from work_dirs/mask_rcnn_swin_tiny_fine_dst_640p3x_1x_coco/epoch_9.pth

sleep 60

# ./dist_train.sh experiments/mask_rcnn_swin_tiny_scratch_dst_224b2_p1x_coco.py 8 #--resume-from work_dirs/mask_rcnn_swin_tiny_patch4_window7_adamw_1x_coco/epoch_7.pth

sleep 36000000
