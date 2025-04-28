import os
from PIL import Image
import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision import datasets, transforms

class SodDataset(data.Dataset):
    """
    SOD dataset
    """
    def __init__(self, data_dir, data_list, image_size, depth_dir=None):
        assert isinstance(image_size, int) or (isinstance(image_size, collections.abc.Iterable) and len(image_size) == 2)
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.data_dir = data_dir
        with open(data_list, 'r') as f:
            self.data_list = [x.strip() for x in f.readlines()]
        self.data_num = len(self.data_list)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
        self.trsfm = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            normalize,
        ])
        
        if depth_dir is not None:
            self.depth_dir = depth_dir
            self.depth_trsfm = transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ])
        else:
            self.depth_dir = None

    def __getitem__(self, item):
        image_name = self.data_list[item]

        image = Image.open(os.path.join(self.data_dir, image_name)).convert('RGB') 
        image_size = torch.Tensor(image.size[::-1])
        image = self.trsfm(image)

        if self.depth_dir is not None:
            depth = Image.open(os.path.join(self.depth_dir, (image_name[:-4] + '.png'))).convert('L')
            depth = self.depth_trsfm(depth)
            return image, image_name, image_size, depth
        
        return image, image_name, image_size

    def __len__(self):
        return self.data_num

def get_loader(data_dir, data_list, image_size=352, batch_size=1, shuffle=False, num_workers=1, pin_memory=False, depth_dir=None):
    dataset = SodDataset(data_dir, data_list, image_size, depth_dir)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, 
                                    shuffle=shuffle, num_workers=num_workers, 
                                    pin_memory=pin_memory)
    return data_loader
