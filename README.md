# End-to-End Human Instance Matting (E2E-HIM)

Official repository for the paper [**End-to-End Human Instance Matting**](https://ieeexplore.ieee.org/document/10224299)

## Description

E2E-HIM is a human instance matting network.

## Demovideo

[![DemoVideo](http://img.youtube.com/vi/WZBqxbE7XU/0.jpg)](https://www.youtube.com/watch?v=-WZBqxbE7XU "Demo Video")
## Requirements
#### Hardware:

GPU memory >= 8GB.

#### Packages:

- torch >= 1.10
- numpy >= 1.16
- opencv-python >= 4.0
- einops >= 0.3.2
- timm >= 0.4.12

## Models
**The model can only be used and distributed for noncommercial purposes.** 

Quantitative results on HIM-100K.

|                                   Model Name                                   |  Size  | EMSE | EMAD |
|:------------------------------------------------------------------------------:|:------:|:----:|:---:|
| [E2E-HIM(Swin-Tiny)](https://pan.baidu.com/s/1dbn_v-qYi8rMN_DrcPUhYA?pwd=klrb) | 154MiB | 5.35 | 6.64 |

## Evaluation
We provide the script `eval_swintiny.py` for evaluation.

## Citation

If you use this model in your research, please cite this project to acknowledge its contribution.

```plaintext
@article{liu2023end,
  title={End-to-end human instance matting},
  author={Liu, Qinglin and Zhang, Shengping and Meng, Quanling and Zhong, Bineng and Liu, Peiqiang and Yao, Hongxun},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2023},
  publisher={IEEE}
}
```
