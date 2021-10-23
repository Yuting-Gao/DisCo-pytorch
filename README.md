# DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning


Pytorch Implementation for [DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning](https://arxiv.org/abs/2104.09124)

### If the project is useful to you, please give us a star. ⭐️

<img width="580" alt="image" src="https://user-images.githubusercontent.com/22510464/124569124-3f0a1800-de78-11eb-8734-dfe86d87197d.png">

```
@article{gao2021disco,
  title={DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning},
  author={Gao, Yuting and Zhuang, Jia-Xin and Li, Ke and Cheng, Hao and Guo, Xiaowei and Huang, Feiyue and Ji, Rongrong and Sun, Xing},
  journal={arXiv preprint arXiv:2104.09124},
  year={2021}
}
```

## Requirements

* Python3
* Pytorch 1.6+
* Detectron2

* 8 GPUs are preferred
* ImageNet, Cifar10/100, VOC, COCO

## Run

Before running, we firstly move all data into share memory

```
cp /path/to/ImageNet /dev/shm
```

### Pretrain Model

For pretraining baseline models with default hidden layer dimension in **Tab1** 

```Python
# Switch to moco directory
cd moco

# R-50
python3 -u main_moco.py -a resnet50 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 2048 /dev/shm/ 2>&1 | tee ./logs/std.log
python3 main_lincls.py -a resnet50 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# R-101
python3 -u main_moco.py -a resnet101 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 2048 /dev/shm/ 2>&1 | tee ./logs/std.log
python3 main_lincls.py -a resnet101 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# R-152
python3 -u main_moco.py -a resnet152 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 800 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 2048 /dev/shm/ 2>&1 | tee ./logs/std.log
python3 main_lincls.py -a resnet152 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0799.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# Mob
python3 -u main_moco.py -a mobilenetv3 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 512 /dev/shm 2>&1 |  tee ./logs/std.log
#          Evaluation
python3 main_lincls.py -a mobilenetv3 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# Effi-B0
python3 -u main_moco.py -a efficientb0 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 1280 2>&1  |  tee ./logs/std.log
#          Evaluation
python3 main_lincls.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# Effi-B1
python3 -u main_moco.py -a efficientb1 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0  --hidden 1280  /dev/shm  2>&1 | tee ./logs/std.log
#          Evaluation
python3 main_lincls.py -a efficientb1 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# R-18
python3 -u main_moco.py -a resnet18 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 1280 /dev/shm/ 2>&1 | tee ./logs/std.log
#          Evaluation
python3 main_lincls.py -a resnet18 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log

# R-34
python3 -u main_moco.py -a resnet34 --batch-size 256 --learning-rate 0.03 --mlp --moco-t 0.2 --aug-plus --cos --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --hidden 1280 /dev/shm/ 2>&1 | tee ./logs/std.log
#          Evaluation
python3 main_lincls.py -a resnet34 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log
```

### DisCo 

For training DisCo in **Tab1**, Comparision with baseline

```python
# Switch to DisCo directory
cd DisCo

# R-50 & Effib0
python3 -u main.py -a efficientb0 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher_arch resnet50 --teacher /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm/ 2>&1 | tee ./logs/std.log
#          Evaluation
python3 -u main_lincls.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log

# R50w2 & Effib0
python3 -u main.py -a efficientb0 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher_arch resnet50w2 --teacher /path/to/swav_RN50w2_400ep_pretrain.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
#          Evaluation
python3 yt_main_lincls.py -a resnet18 --learning-rate 30.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar  /dev/shm 2>&1 | tee ./logs/std.log
```

For **Tab2**, Linear evaluation top-1 accuracy (%) on ImageNet compared with **different distillation methods.**

```python
# RKD+DisCo, Eff-b0
python3 -u main_moco_distill_rkd.py -a efficientb0 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher /path/to/teacher_res50.pth.tar --use-mse /dev/shm  2>&1 | tee ./logs/std.log
#                  Evaluation
python3 -u main_lincls.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log

# RKD, Eff-b0
python3 -u main_moco_distill_rkd.py -a efficientb0 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher /path/to/teacher_res50.pth.tar /dev/shm  2>&1 | tee ./logs/std.log
#                  Evaluation
python3 -u main_lincls.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
```

For **Tab3** , **Object detection and instance segmentation results **

```python
# Cp data to /dev/shm and set up path for Detectron2
cp -r /path/to/VOCdevkit/* /dev/shm/
cp -r /path/to/coco_2017 /dev/shm/coco
export DETECTRON2_DATASETS=/dev/shm

pip install /youtu-reid/jiaxzhuang/acmm/detectron2-0.4+cu101-cp36-cp36m-linux_x86_64.whl
cd detection

# Convert model for Detectron2
python3 convert-pretrain-to-detectron2.py /path/ckpt/checkpoint_0199.pth.tar ./output.pkl

# Evaluation on VOC
python3 train_net.py --config-file configs/pascal_voc_R_50_C4_24k_moco.yaml --num-gpus 8 --resume MODEL.RESNETS.DEPTH 34 MODEL.RESNETS.RES2_OUT_CHANNELS 64 2>&1 | tee ../logs/std.log
# Evaluation on CoCo
python3 train_net.py --config-file configs/coco_R_50_C4_2x_moco.yaml --num-gpus 8  --resume MODEL.RESNETS.DEPTH 18 MODEL.RESNETS.RES2_OUT_CHANNELS 64 2>&1 | tee ../logs/std.log
```



For  **Fig5** , evaluation on **Semi-Supervised Tasks** 

```python
# Copy 1%, 10% ImageNet from the complete ImageNet, according to split from SimCLR.
cd data
# Need to set up path to Compelete ImageNet and the output path.
python3 -u imagenet_1_fraction.py --ratio 1
python3 -u imagenet_1_fraction.py --ratio 10

# Evaluation on 1% ImageNet with Eff-B0 by DisCo
cp -r /path/to/imagenet_1_fraction/train  /dev/shm
cp -r /path/to/imagenet_1_fraction/val  /dev/shm/
python3 -u main_lincls_semi.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm  2>&1 | tee ./logs/std.log

# Evaluation on 10% ImageNet with R-18 by DisCo
cp -r /path/to/imagenet_10_fraction/train  /dev/shm
cp -r /path/to/imagenet_10_fraction/val  /dev/shm/
python3 -u main_lincls_semi.py -a resnet18 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm  2>&1 | tee ./logs/std.log
```

For Fig6, evaluation on **Cifar10/Cifar100**

```python
# Copy Cifar10/100 to /dev/shm
cp /path/to/Cifar10/100 /dev/shm

# Evaluation on 1% Cifar10 with Eff-B0 by DisCo
python3 cifar_main_lincls.py -a efficientb0 --dataset cifar10 --lr 3 --epochs 200 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
# Evaluation on  Cifar100 with Resnet18 by DisCo
python3 cifar_main_lincls.py -a resnet18 --dataset cifar100 --lr 3 --epochs 200 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
```

For  **Tab4**, Linear evaluation top-1 accuracy (%) on ImageNet, compared with **SEED with consistent dimension in hidden layer**.

```Python
python3 -u main.py -a efficientb0 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher_arch resnet50 --teacher /path/to/ckpt/checkpoint_0199.pth.tar --hidden 2048 /dev/shm/ 2>&1 | tee ./logs/std.log
#          Evaluation
python3 -u main_lincls.py -a efficientb0 --learning-rate 3.0 --batch-size 256 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained /path/to/ckpt/checkpoint_0199.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
```

For **Tab5**, Linear evaluation top-1 accuracy (%) on ImageNet with SwAV as the testbed.

```Python
# SwAV, Train with SwAV only
cd swav-master
python3 -m torch.distributed.launch --nproc_per_node=8 main_swav.py \
        --data_path /dev/shm/train \
        --base_lr 0.6 \
        --final_lr 0.0006 \
        --warmup_epochs 0 \
        --crops_for_assign 0 1 \
        --size_crops 224 96 \
        --nmb_crops 2 6 \
        --min_scale_crops 0.14 0.05 \
        --max_scale_crops 1. 0.14 \
        --use_fp16 true \
        --freeze_prototypes_niters 5005 \
        --queue_length 3840 \
        --epoch_queue_starts 15 \
        --dump_path ./ckpt \
        --sync_bn pytorch \
        --temperature 0.1 \
        --epsilon 0.05 \
        --sinkhorn_iterations 3 \
        --feat_dim 128 \
        --nmb_prototypes 3000 \
        --epochs 200 \
        --batch_size 64 \
        --wd 0.000001 \
        --arch efficientb0 \
        --use_fp16 true 2>&1 | tee ./logs/std.log
# Evaluation
python3 -m torch.distributed.launch --nproc_per_node=8 eval_linear.py --arch efficientb0 --data_path /dev/shm --pretrained /path/to/ckpt/checkpoints/ckp-199.pth 2>&1 | tee ./logs/std.log

# DisCo + SwAV
python3 -m torch.distributed.launch --nproc_per_node=8 main_swav_distill.py \
        --data_path /dev/shm/train \
        --base_lr 0.6 \
        --final_lr 0.0006 \
        --warmup_epochs 0 \
        --crops_for_assign 0 1 \
        --size_crops 224 96 \
        --nmb_crops 2 6 \
        --min_scale_crops 0.14 0.05 \
        --max_scale_crops 1. 0.14 \
        --use_fp16 true \
        --freeze_prototypes_niters 5005 \
        --queue_length 3840 \
        --epoch_queue_starts 15 \
        --dump_path ./ckpt \
        --sync_bn pytorch \
        --temperature 0.1 \
        --epsilon 0.05 \
        --sinkhorn_iterations 3 \
        --feat_dim 128 \
        --nmb_prototypes 3000 \
        --epochs 200 \
        --batch_size 64 \
        --wd 0.000001 \
        --arch efficientb0 \
        --pretrained /path/to/swav_800ep_pretrain.pth.tar 2>&1 | tee ./logs/std.log
```

For **Tab6**, Linear evaluation top-1 accuracy (%) on ImageNet with variants of teacher pre-training methods.

```Python

# SwAV
python3 -u main.py -a resnet34 --lr 0.03 --batch-size 256 --moco-t 0.2 --aug-plus --dist-url 'tcp://localhost:10043' --multiprocessing-distributed --world-size 1 --rank 0 --mlp --cos --teacher_arch SWAVresnet50 --teacher /path/to/swav_800ep_pretrain.pth.tar /dev/shm 2>&1 | tee ./logs/std.log
```

### Visualization

```
cd DisCo
# Generate Embed
# Move Embed to data path

python -u draw.py
```

### BI Analysis

code for BI is put on **data**

------



## Checkpoints

TO BE Releasese...

| Architecture | Self-supervised Methods | Model Checkpoints                                            |
| :----------- | ----------------------- | ------------------------------------------------------------ |
| ResNet152    | MoCo-V2                 | [Model](https://drive.google.com/file/d/1HwBJG16zCIQ1-ILa7cvGEAYaKlkWK3mG/view?usp=sharing) |
| ResNet101    | MoCo-V2                 | [Model](https://drive.google.com/file/d/1gi6_qbr921hnyth6RIkZtzQOp8IYZ5Tb/view?usp=sharing) |
| ResNet50     | MoCo-V2                 | [Model](https://drive.google.com/file/d/10eDoXeDgK4MlfjDDbV1R7n3uSPlzs-1q/view?usp=sharing) |

For teacher models such as ResNet-50*2 etc, we use their official implementation, which can be downloaded from their github pages. 



## Thanks

Code heavily depends on MoCo-V2, Detectron2.
