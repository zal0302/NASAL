import os
import torch
import torch.nn.functional as F
from networks.NASAL import get_model
import numpy as np
from PIL import Image
from flops import simplesum

class Solver(object):
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config
        self.net = get_model(self.config.net_config)
        self.device = torch.device('cuda' if (torch.cuda.is_available() and self.config.cuda) else 'cpu')
        self.net = self.net.to(self.device)
        print('Loading pre-trained model from %s...' % self.config.model)
        self.net.load_state_dict(torch.load(self.config.model)['model'])
        self.net.eval()

    def test(self):
        print('Begin testing ...')
        with torch.no_grad():
            if self.config.input_type == 'rgb':
                for i, (data, image_name, image_size) in enumerate(self.data_loader):
                    data = data.to(self.device).float()
                    output = self.net(data)

                    for output_, image_name_, image_size_ in zip(output, image_name, image_size):
                        if output_.shape[-2:] != image_size_:
                            output_ = F.interpolate(output_.unsqueeze(0), tuple(image_size_.int().tolist()), mode='bilinear', align_corners=True)
                        output_ = np.squeeze(torch.sigmoid(output_).cpu().data.numpy())

                        Image.fromarray(output_ * 255).convert("L").save(os.path.join(self.config.test_fold, image_name_[:-4] + '.png'))
            elif self.config.input_type == 'rgbd':
                for i, (data, image_name, image_size, depth) in enumerate(self.data_loader):
                    data = data.to(self.device).float()
                    depth = depth.to(self.device).float()
                    output = self.net(data, depth)

                    for output_, image_name_, image_size_ in zip(output, image_name, image_size):
                        if output_.shape[-2:] != image_size_:
                            output_ = F.interpolate(output_.unsqueeze(0), tuple(image_size_.int().tolist()), mode='bilinear', align_corners=True)
                        output_ = np.squeeze(torch.sigmoid(output_).cpu().data.numpy())

                        Image.fromarray(output_ * 255).convert("L").save(os.path.join(self.config.test_fold, image_name_[:-4] + '.png'))
        print('Testing finished. The predictions are saved under %s' % self.config.test_fold)
        self.net.cpu()
        # prams, flops = simplesum(self.net, inputsize=(3, 320, 320), device=-1)
    
    





