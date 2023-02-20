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
| ResNet152    | MoCo-V2                 | [ResNet152-checkpoint_0799.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/Ea-cJlU8dV1IhyxUqfvqNFwBnhf3y7P9h0KBjK8eUluboQ?e=jKUIOJ) |
| ResNet101    | MoCo-V2                 | [ResNet101-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EYwQUxw76pRGodADpRZK42AB7j1iRu0VTDz7OgZyRwaJ2A?e=E9Cg20) |
| ResNet50     | MoCo-V2                 | [ResNet50-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EZaGJw0bXsFAp5JUTzhPtecBeyk77qOPqvG5E24H9Pehmg?e=9wtpzJ) |

For teacher models such as ResNet-50*2 etc, we use their official implementation, which can be downloaded from their github pages. 

### Student Models by DisCo

| Teacher/Students | Efficient-B0   |   Efficient-B1   | ResNet-18  | ResNet-34 | MobileNet-v3 |Vit-Tiny   | XCiT-Tiny                                                   |
| ----------------| --------------------------- | --------------------------- | --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| ResNet-50        | [ResNet50-EfficientB0-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EZ3HDEiMWN9Hh8q2Vdk70vABCNaEKahc4-G2lo6n-hm_UA?e=dagoX8) | [DisCo-R50-Effb1.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EV1WPVvdupJFhf8dIt5UHlMBA1ISoPvmS5i5HKAfWOjMLg?e=LRMvua) | [ResNet50-ResNet18-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/ET_i6g883Z5Gk8_WMRRypsIBvsDFCkoEHNaX1fPQRvhQ8A?e=SGeALD) | [DisCo-R50-R34.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EZVX5NO2y69NoCe3BI-PNF0BpKNJR1QHKKNZsDfXcTZ0yw?e=CWsuv3) | [DisCo-R50-Mob.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/ERiHbUJATK9GlIa2wJZmcqMBs3sHHzVzSrnyEPNBwNp0rA?e=WiGbYW) |-                                                            | -                                                            |
| ResNet-101       | [ResNet101-EfficientB0-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EUFnxWihEf9DgsJF0qXqQqwBaGz0COwJEpgcq_QY141MQg?e=PLdWgX) | [DisCo-R101-Effb1.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EekuIGR6ridFvAg3NhD9GxQBqpMfjucXIjjHSSdffkPn-w?e=7bqAf2) | [ResNet101-ResNet18-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/ERAE008V4TFForMi1C-oNmEBFpgZQmvUJmVNFENC2WXD6Q?e=Fmkptt) | [DisCo-R101-R34.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EYCLWUYDqhhOiEa6ds5rjHYBKS7GRrEHToGqsbwBLf_jug?e=CmOOdM) |[DisCo-R101-Mob.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/ERu92035ZmNHk9lvuyj6dGIBVKZ_SO1r6ZoR_Ca_AZ207A?e=76VvCt)|-                                                            | -                                                            |
| ResNet-152       | [ResNet152-EfficientB0-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EUwOfm8UmNtJssIShAZIiLsBnbJGY8VCTcYBXOuVdB-b4w?e=O7G897) | [DisCo-R152-Effb1.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EboHOWOGpvNAvdXH9mAef1wBlwY6MyoG6X1ERKi8TUAbYQ?e=a5UmJs) | [ResNet152-ResNet18-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EbrXFWngADdNtFMDm1d6ScEB3AjbMxbQ4AxG3RDSLR2ecg?e=SMfbsa) | [DisCo-R152-R34.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/Eb-zmWEJm75Hp6-M9glXGH8BCJHimbWfi5GDV1BYUoLXhA?e=QvZ9yO) | [DisCo-R152-Mob.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/Eb2v_D4w8NlDh0cT9W6P2wsBeS7aL4y1wiRzDsoVbjRpew?e=ISK4bF) |-                                                            | -                                                            |
| ResNet-50*2      | [ResNet50w2-EfficientB0-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/ETrIFUa1c_RIvYOSb_eiQMMBDr0uO6xMVRBvbvOfKP40fg?e=cuJlNv) | [DisCo-RN50x2-Effb1.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EUfm_r2MdFtItmfCiwbjtRABlHkn6KchK0DW_0Qyi_9VQA?e=rfOpKu) | [ResNet50w2-ResNet18-checkpoint_0199.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EVigZpNTekJHi1ltj-uKPagB_HvWSStRuAPHk1wAfaa-eg?e=CPrD5M) | [DisCo-RN50x2-R34.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EZR0Xt7hVwFAh3r1bH5A86oB0mNMv82KZDoBgrk1x1IJFw?e=esZF0r) | [DisCo-RN50x2-Mob.pth.tar](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/Ef41UHaon0tFq1kDImaOXOYBiglIOgEoCb6acpWuXIp2Fg?e=mcHkvE) |-                                                            | -                                                            |
| ViT-Small        | -                                                            | -                                                            | -                                                            |-                                                            |-                                                            | [DisCo-ViT-small-ViT-tiny.pth](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/Ebsv3xQse-tHgvqzogyGPo8BvzmoJlx3_Ua3PTNpooiT_g?e=ExIgjU) | -                                                            |
| XCiT-Small       | -                                                            | -                                                            | -                                                            | -                                                            |-                                                            | -                                                            | [XCiT-Small-XCiT-Tiny-checkpoint.pth](https://hkustconnect-my.sharepoint.com/:u:/g/personal/jzhuangad_connect_ust_hk/EaE_Zp4lj55ClAEV2ByqzrcBspUO0fSWQgDo008zz6WQtw?e=x3Ryvy) |



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
