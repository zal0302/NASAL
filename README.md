## Towards Efficient Salient Object Detection via U-Shape Architecture Search

### This is the official demo PyTorch implementation of our KBS 2025 [paper](https://www.sciencedirect.com/science/article/pii/S0950705125005611).

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)


## Demo usage
1. Move to the home folder.
```shell
cd NASAL/
```

2. The source images are in the `demo/rgb_img` and `demo/rgbd_img` folders.


By running 
```shell
sh test_rgb.sh
```
you'll get the RGB SOD predictions under
the `demo/rgb_pre` folder. 

By running 
```shell
sh test_rgbd.sh
```
you'll get the RGB-D SOD predictions under
the `demo/rgbd_pre` folder. 


## Pre-computed results and evaluation results

You can find the pre-computed prediction maps on all the datasets reported in the paper and their corresponding evaluation scores with the following link:
[Results reported in the paper](https://drive.google.com/file/d/1HjKIMjTMhzXL7UkX6gGga_BioTCwJUtL/view?usp=sharing)

## If you think this work is helpful, please cite

```latex
@article{liu2025towards,
title = {Towards efficient salient object detection via U-shape architecture search},
journal = {Knowledge-Based Systems},
volume = {318},
pages = {113515},
year = {2025},
issn = {0950-7051},
doi = {https://doi.org/10.1016/j.knosys.2025.113515},
author = {Zhi-Ang Liu and Jiang-Jiang Liu}
}
```
