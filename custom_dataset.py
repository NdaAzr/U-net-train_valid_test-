# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 17:49:08 2018

@author: Neda
"""

#conda update spyder

import torch
from torch.utils.data.dataset import Dataset  # For custom data-sets
import torchvision.transforms as transforms
from PIL import Image
import numpy 
   
        
class CustomDataset(Dataset):
    def __init__(self, image_paths, target_paths):   # initial logic happens like transform

        self.image_paths = image_paths
        self.target_paths = target_paths
        self.transforms = transforms.ToTensor()
        self.mapping = {
            0: 0,
            255: 1              
        }
        
    def mask_to_class(self, mask):
        for k in self.mapping:
            mask[mask==k] = self.mapping[k]
        return mask
    
    def __getitem__(self, index):

        image = Image.open(self.image_paths[index])
        mask = Image.open(self.target_paths[index])
        t_image = image.convert('L')
        t_image = self.transforms(t_image)
        #mask = torch.from_numpy(np.array(mask))    #this is for BMCC dataset
        mask = torch.from_numpy(numpy.array(mask, dtype=numpy.uint8)) # this is for my dataset(lv)
        mask = self.mask_to_class(mask)
        mask = mask.long()
        return t_image, mask, self.image_paths[index], self.target_paths[index] 
    
    def __len__(self):  # return count of sample we have

        return len(self.image_paths)

  