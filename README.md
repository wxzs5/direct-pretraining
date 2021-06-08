# Rethinking Training from Scratch for Object Detection

## Intro
Code for paper [Rethinking Training from Scratch for Object Detection](https://arxiv.org/abs/2106.03112).

The ImageNet pre-training initialization is the de-facto standard for object detection. He et al. found it is possible to train detector from scratch(random initialization) while needing a longer training schedule with proper normalization technique. In this paper, we explore to directly pre-training on target dataset for object detection. Under this situation, we discover that the widely adopted large resizing strategy e.g. resize image to (1333, 800) is important for fine-tuning but it's not necessary for pre-training. Specifically, we propose a new training pipeline for object detection that follows `pre-training and fine-tuning', utilizing low resolution images within target dataset to pre-training detector then load it to fine-tuning with high resolution images. With this strategy, we can use batch normalization(BN) with large bath size during pre-training, it's also memory efficient that we can apply it on machine with very limited GPU memory(11G). We call it direct detection pre-training, and also use direct pre-training for short. Experiment results show that direct pre-training accelerates the pre-training phase by more than 11x on COCO dataset while with even +1.8mAP compared to ImageNet pre-training. Besides, we found direct pre-training is also applicable to transformer based backbones e.g. Swin Transformer.

![](figs/process.png)


## Pre-trained models 

| method            | pipeline       | bbox mAP | mask mAP | config                                                                                                                                                                                       | model                                                                                                                                                                                                                                                                                  |
| ----------------- | -------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| RetinaNet         | ImageNet-1x    | 36.5     | -        | -                                                                                                                                                                                            | -                                                                                                                                                                                                                                                                                      |
| RetinaNet         | Direct(P1x)-1x | **37.1** | -        | [pre-train](./configs/direct_pretraining/retinanet_r50_fpn_direct_pretrain_1x_coco.py)&#124;[fine-tune](./configs/direct_pretraining/retinanet_r50_fpn_direct_finetune_p1x_1x_coco.py)       | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EcLiASQSUKlBpsY15lacsgoBWfARylTODd7hNVD2t0Pjgw)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EcCAmSBF3VhGtmsChX8MHG8BDKj4KGu0zGNuyFzs325lMw) |
|                   |                |          |          |                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                        |
| Faster RCNN       | ImageNet-1x    | 37.4     | -        | -                                                                                                                                                                                            | -                                                                                                                                                                                                                                                                                      |
| Faster RCNN       | Direct(P1x)-1x | **39.3** | -        | [pre-train](./configs/direct_pretraining/faster_rcnn_r50_fpn_direct_pretrain_1x_coco.py)&#124;[fine-tune](./configs/direct_pretraining/faster_rcnn_r50_fpn_direct_finetune_p1x_1x_coco.py)   | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EcbSrMe0c5BBm-TaiKVyXwEB2k82ytg4RnaLJuKTNrZnzg)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/ERRztLC6Tv1AiLd5JcdCbxQBF4U9ieUkLmzPWJdHhHc-pg) |
|                   |                |          |          |                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                        |
| Cascade RCNN      | ImageNet-1x    | 40.3     | -        | -                                                                                                                                                                                            | -                                                                                                                                                                                                                                                                                      |
| Cascade RCNN      | Direct(P1x)-1x | **41.5** | -        | [pre-train](./configs/direct_pretraining/cascade_rcnn_r50_fpn_direct_pretrain_1x_coco.py)&#124;[fine-tune](./configs/direct_pretraining/cascade_rcnn_r50_fpn_direct_finetune_p1x_1x_coco.py) | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EVSJBWfx8fJLnN7u4ZrdoXgBXTMsIJQJVUw9DNmL0_uH8Q)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/ESC2eKKdaRJEhZpycyLAQxkBwC6A37er-kF5nWUPmd_qsQ) |
|                   |                |          |          |                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                        |
| Mask RCNN         | ImageNet-1x    | 38.2     | 34.7     | -                                                                                                                                                                                            | -                                                                                                                                                                                                                                                                                      |
| Mask RCNN         | Direct(P1x)-1x | **40.0** | **35.8** | [pre-train](./configs/direct_pretraining/mask_rcnn_r50_fpn_direct_pretrain_1x_coco.py)&#124;[fine-tune](./configs/direct_pretraining/mask_rcnn_r50_fpn_direct_finetune_p1x_1x_coco.py)       | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EfFw3sRNLHhKm6eeWmeILXYBdEyd6iF5JDM38kM9Pomc0A)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EVceRaCRC-lFvjdTCg1EfoAB099qFBvwxTidb4NkrhSLEA) |
|                   |                |          |          |                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                        |
| Mask RCNN w/ Swin | ImageNet-1x    | 43.8     | 39.6     | -                                                                                                                                                                                            | -                                                                                                                                                                                                                                                                                      |
| Mask RCNN w/ Swin | Direct(P1x)-1x | **45.0** | **40.5** | [pre-train](./configs/direct_pretraining/mask_rcnn_swin_tiny_direct_pretrain_1x_coco.py)&#124;[fine-tune](./configs/direct_pretraining/mask_rcnn_swin_tiny_direct_finetune_p1x_1x_coco.py)   | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EY5yWXpfGCRNnB93MwSucUABy7lqFEK5b8qwCXq6NFaO4w)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EdyFCPCDGxlFlJSM66rG93gBNGLdLePin2O3oAgPTtEMBw) |


We also provide models are sufficiently trained with longer schedule(3x), could be used for model initialization.

| method            | pipeline       | bbox mAP | mask mAP | model                                                                                                                                                                                                                                                                                  |
| ----------------- | -------------- | -------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Mask RCNN         | Direct(P3x)-1x | **41.5** | **37.0** | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EXhGvWfO5GNOgK8liP7dpQUB1Ly4QcDGdh6mNYxGoQkV-w)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EUuTtKN407BAiY5ns5Pfv3IBkMazJj0oYyRLSHR6SyTQTw) |
| Mask RCNN w/ Swin | Direct(P3x)-1x | **46.9** | **41.9** | [pre-train](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/ETsM3wbvzFpItXK-tgqgBssBW3q6PDyzlMqIp5HrZ2i72A)&#124;[fine-tune](https://dutaotbsteduau-my.sharepoint.com/:u:/g/personal/love0735_edu365_site/EUINYgICG8hAoNZ4YgV6Hl4Bb28YjNo7CmvGw_FMzIbylw) |

## Installation

This project mainly reference [MMDetection](https://github.com/open-mmlab/mmdetection) codebase and [Swin Transformer](https://github.com/SwinTransformer/Swin-Transformer-Object-Detection). 

- Following [MMDection installation guide](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md) to install mmdetection.
- install [NVIDIA apex](https://github.com/NVIDIA/apex):
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```

## Usage

### Prepare Dataset

```bash
git clone https://github.com/wxzs5/direct-pretraining.git
cd direct-pretraining

# set dataset path
mkdir data
ln -s path/to/coco_dataset data/

```

### Inference
```bash
# single-gpu testing
python test.py <CONFIG_FILE> <DET_CHECKPOINT_FILE> --eval <EVAL_METRICS>

# multi-gpu testing
./dist_test.sh <CONFIG_FILE> <DET_CHECKPOINT_FILE> <GPU_NUM> --eval <EVAL_METRICS>
```

e.g. 
```bash
./dist_test.sh configs/direct_pretraining/mask_rcnn_r50_fpn_direct_finetune_p1x_1x_coco.py model.pth 8 --eval bbox segm
```

### Train

- pre-training:
```bash
./dist_train.sh <CONFIG_FILE> <GPU_NUM>
```
- fine-tuning:  
```bash
# set config file *load_from* the pre-trained model path
./dist_train.sh <CONFIG_FILE> <GPU_NUM>
```
## Citing Direct Pre-training

if this paper helps you, please consider citing direct pre-training.
```bibtex
@article{yang2021Rethink,
  title={Rethinking Training from Scratch for Object Detection},
  author={Yang, Li and Hong, Zhang and Yu, Zhang},
  journal={arXiv preprint arXiv:2106.03112},
  year={2021}
}
```