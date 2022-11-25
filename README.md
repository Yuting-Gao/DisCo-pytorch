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

| Teacher/Students | Efficient-B0   |   Efficient-B1   | ResNet-18  | ResNet-34 | MobileNet-v3-Large |Vit-Tiny   | XCiT-Tiny                                                   |
| ----------------| --------------------------- | --------------------------- | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet-50        | [Model](https://drive.google.com/file/d/1poiuaKdezRuhmOprA-kP2hNRyWEeYaZI/view?usp=sharing) | [Model](https://drive.google.com/file/d/1IXEGljlbn7Jt1dcjpnT9-9oKX0rXkvWa/view?usp=sharing) | [Model](https://drive.google.com/file/d/10Ry3OPGsc_pS6LGh03eJ2ASO26_utncG/view?usp=sharing) | [Model](https://drive.google.com/file/d/1XuVT575g-hsg-wuvyqxcS02j-7zQsU3G/view?usp=sharing) | [Model](https://drive.google.com/file/d/1Kv8D3WqVWbajReATP0Rqp1BXSf0q8xiU/view?usp=sharing) |-                                                            | -                                                            ||
| ResNet-101       | [Model](https://drive.google.com/file/d/1XjwWiw_IXgOIxQKrPK5wQSgSlr8XsHzl/view?usp=sharing) | [Model](https://drive.google.com/file/d/1rgbU317OovdFjSXqDEjAB6SFj95gWOWS/view?usp=sharing) | [Model](https://drive.google.com/file/d/103NHdXrLi7my1cB9aQR8BR7fcn-D89zi/view?usp=sharing) | [Model](https://drive.google.com/file/d/1HTF-6p6Sj0B4H8UYC6NJAKYXbPkoiUDd/view?usp=sharing)|[Model](https://drive.google.com/file/d/1qc5DcHXo_BFHsblcIgMn3KBXIAZUOZwN/view?usp=sharing)|-                                                            | -                                                            |
| ResNet-152       | [Model](https://drive.google.com/file/d/1XjwWiw_IXgOIxQKrPK5wQSgSlr8XsHzl/view?usp=sharing) | [Model](https://drive.google.com/file/d/1r3o_mL1ETC-jlIjIBPYfXJ_bxpRQ1qHB/view?usp=sharing) | [Model](https://drive.google.com/file/d/1AqZJ8iJPDkLgRbvOFUXySnk3ZVJVqKHX/view?usp=sharing) | [Model](https://drive.google.com/file/d/14bfR6Tjk_eSMG72vnP6QYAiMqVNw0m1c/view?usp=sharing) | [Model](https://drive.google.com/file/d/1x1UdcYFxbnfn-TpDyhmZsJupz3-5rWKm/view?usp=sharing) |-                                                            | -                                                            |
| ResNet-50*2      | [Model](https://drive.google.com/file/d/1ZxnmazOZ90POpj_1ynrDvI_2kF6mXyoe/view?usp=sharing) | [Model](https://drive.google.com/file/d/1CpW2ZP_HeFgVaFNP4Ne96eSPMdFrsDsc/view?usp=sharing) | [Model](https://drive.google.com/file/d/15s3fbwD8u0kceEO9Nu158xeb-fRY9h5R/view?usp=sharing) | [Model](https://drive.google.com/file/d/1THz_B0rdtSx5J-Ifo-9sW9-qI2eNXPCz/view?usp=sharing) | [Model](https://drive.google.com/file/d/1xFE_ds6aesMP7-BUDxUSONPwOyQnYB9k/view?usp=share_link) |-                                                            | -                                                            |
| ViT-Small        | -                                                            | -                                                            | -                                                            |-                                                            |-                                                            | [Model](https://drive.google.com/file/d/1wgswIc_7LLyEjmXha5PGfihDbsyzIZqy/view?usp=share_link) | -                                                            |
| XCiT-Small       | -                                                            | -                                                            | -                                                            | -                                                            |-                                                            | -                                                            | [Model](https://drive.google.com/file/d/1fxaqR-diiZ5ufQx5f0cZ1nJKCa1CCo_1/view?usp=sharing) |



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
