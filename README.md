## NASAL: Salient Object Detection via Searching Efficient U-shape Structures

### This is a demo PyTorch implementation.

## Prerequisites

- [Pytorch 0.4.1+](http://pytorch.org/)
- [torchvision](http://pytorch.org/)


## Demo usage
1. Move to the home folder.
```shell
cd NASAL_demo/
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

You can find the pre-computed predictions maps on all the five datasets reported in the paper and their corresponding evaluation scores with the following link:
[Results reported in the paper](https://drive.google.com/file/d/1HjKIMjTMhzXL7UkX6gGga_BioTCwJUtL/view?usp=sharing)