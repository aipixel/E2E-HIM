# End-to-End Human Instance Matting (E2E-HIM)

Official repository for the paper [**End-to-End Human Instance Matting**](https://ieeexplore.ieee.org/document/10224299)

## Description

E2E-HIM is a human instance matting network.

## Demovideo

[![DemoVideo](http://img.youtube.com/vi/-WZBqxbE7XU/0.jpg)](https://www.youtube.com/watch?v=-WZBqxbE7XU "Demo Video")

## Requirements
#### Hardware:

GPU memory >= 6GB.

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
|:------------------------------------------------------------------------------:|:------:|:----:|:----:|
| [E2E-HIM](https://mega.nz/file/3IJAFDYQ#FFJk7FADXqYjr-FHDmmW6MbGPf5TVvYocID3RaVaa28) | 270MiB | 5.33 | 6.62 |

## Evaluation
We provide the script `eval_swintiny.py` for evaluation. Note that, current E2E-HIM cannot be applied to high-resolution images.

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

