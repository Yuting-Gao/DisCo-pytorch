# DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning(ECCV-2022 Oral)



This repository contains the **Official Pytorch Implementation** for [DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning](https://arxiv.org/abs/2104.09124)

```
@article{gao2021disco,
  title={DisCo: Remedy Self-supervised Learning on Lightweight Models with Distilled Contrastive Learning},
  author={Yuting Gao, Jia-Xin Zhuang, Shaohui Lin, Hao Cheng, Xing Sun, Ke Li, Chunhua Shen},
  journal={European Conference on Computer Vision(ECCV)},
  year={2022}
}
```

If the project is useful to you, please give us a star. ⭐️


## Framework

<img width="580" alt="image" src="https://user-images.githubusercontent.com/22510464/124569124-3f0a1800-de78-11eb-8734-dfe86d87197d.png">


## Checkpoints

### Teacher Models 

| Architecture | Self-supervised Methods | Model Checkpoints                                            |
| :----------- | ----------------------- | ------------------------------------------------------------ |
| ResNet152    | MoCo-V2                 | [Model](https://drive.google.com/file/d/1HwBJG16zCIQ1-ILa7cvGEAYaKlkWK3mG/view?usp=sharing) |
| ResNet101    | MoCo-V2                 | [Model](https://drive.google.com/file/d/1gi6_qbr921hnyth6RIkZtzQOp8IYZ5Tb/view?usp=sharing) |
| ResNet50     | MoCo-V2                 | [Model](https://drive.google.com/file/d/10eDoXeDgK4MlfjDDbV1R7n3uSPlzs-1q/view?usp=sharing) |

For teacher models such as ResNet-50*2 etc, we use their official implementation, which can be downloaded from their github pages. 

### Student Models by DisCo

| Teacher/Students | Efficient-B0                                                 | ResNet-18                                                    | Vit-Tiny                                                     | XCiT-Tiny                                                    |
| ---------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet-50        | [Model](https://drive.google.com/file/d/1poiuaKdezRuhmOprA-kP2hNRyWEeYaZI/view?usp=sharing) | [Model](https://drive.google.com/file/d/10Ry3OPGsc_pS6LGh03eJ2ASO26_utncG/view?usp=sharing) | -                                                            | -                                                            |
| ResNet-101       | [Model](https://drive.google.com/file/d/1XjwWiw_IXgOIxQKrPK5wQSgSlr8XsHzl/view?usp=sharing) | [Model](https://drive.google.com/file/d/103NHdXrLi7my1cB9aQR8BR7fcn-D89zi/view?usp=sharing) | -                                                            | -                                                            |
| ResNet-152       | [Model](https://drive.google.com/file/d/1XjwWiw_IXgOIxQKrPK5wQSgSlr8XsHzl/view?usp=sharing) | [Model](https://drive.google.com/file/d/1AqZJ8iJPDkLgRbvOFUXySnk3ZVJVqKHX/view?usp=sharing) | -                                                            | -                                                            |
| ResNet-50*2      | [Model](https://drive.google.com/file/d/1ZxnmazOZ90POpj_1ynrDvI_2kF6mXyoe/view?usp=sharing) | [Model](https://drive.google.com/file/d/15s3fbwD8u0kceEO9Nu158xeb-fRY9h5R/view?usp=sharing) | -                                                            | -                                                            |
| ViT-Small        | -                                                            | -                                                            | [Model](https://drive.google.com/file/d/1_Im5Vfdl0Q9KhO_W46WKbMDMSMKvWAr0/view?usp=sharing) | -                                                            |
| XCiT-Small       | -                                                            | -                                                            | -                                                            | [Model](https://drive.google.com/file/d/1fxaqR-diiZ5ufQx5f0cZ1nJKCa1CCo_1/view?usp=sharing) |



## Requirements

* Python3
* Pytorch 1.6+
* Detectron2

* 8 GPUs are preferred
* ImageNet, Cifar10/100, VOC, COCO


## Reproduction
Commands can be found on [Reproduction](./Reproduction.md).

## Thanks
Code heavily depends on MoCo-V2, Detectron2.
