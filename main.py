import os 
import argparse
from dataset.dataset import get_loader
from solver import Solver

def main(config):
    test_loader = get_loader(config.data_dir, config.data_list, config.image_size, depth_dir=config.depth_dir)
    if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
    test = Solver(test_loader, config)
    test.test()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--image_size', type=int, default=320)
    
    # Testing for RGB SOD
    parser.add_argument('--input_type', type=str, default='rgb')
    parser.add_argument('--model', type=str, default='pretrained/NASAL.pth.tar')
    parser.add_argument('--net_config', type=str, default='pretrained/NASAL.config')
    parser.add_argument('--test_fold', type=str, default='demo/rgb_pre')
    parser.add_argument('--data_dir', type=str, default='demo/rgb_img')
    parser.add_argument('--data_list', type=str, default='demo/rgb.lst')
    parser.add_argument('--depth_dir', type=str, default=None)

    # Testing for RGB-D SOD
    # parser.add_argument('--input_type', type=str, default='rgbd')
    # parser.add_argument('--model', type=str, default='pretrained/NASAL-D.pth.tar')
    # parser.add_argument('--net_config', type=str, default='pretrained/NASAL-D.config')
    # parser.add_argument('--test_fold', type=str, default='demo/rgbd_pre')
    # parser.add_argument('--data_dir', type=str, default='demo/rgbd_img')
    # parser.add_argument('--data_list', type=str, default='demo/rgbd.lst')
    # parser.add_argument('--depth_dir', type=str, default='demo/depth')
     
    config = parser.parse_args()
    main(config)
