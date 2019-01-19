# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 18:22:06 2019

@author: Neda
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 18:33:34 2018

@author: Neda
"""
from custom_dataset import CustomDataset
import torch
import glob


# get all the image and mask path and number of images

folder_data = glob.glob("D:\\Neda\\Pytorch\\U-net\\my_data\\imagesResized\\*.png")
folder_mask = glob.glob("D:\\Neda\\Pytorch\\U-net\\my_data\\labelsResized\\*.png")

# split these path using a certain percentage
len_data = len(folder_data)
print("count of dataset: ", len_data)
# count of dataset:  992

split_1 = int(0.8 * len(folder_data))
split_2 = int(0.9 * len(folder_data))

folder_data.sort()

train_image_paths = folder_data[:split_1]
print("count of train images is: ", len(train_image_paths)) 
#count of train images is:  793

valid_image_paths = folder_data[split_1:split_2]
print("count of validation image is: ", len(valid_image_paths))
#count of validation image is:  99

test_image_paths = folder_data[split_2:]
print("count of test images is: ", len(test_image_paths)) 
#count of test images is:  100


#print(test_image_paths)

train_mask_paths = folder_mask[:split_1]

valid_mask_paths = folder_mask[split_1:split_2]

test_mask_paths = folder_mask[split_2:]


train_dataset = CustomDataset(train_image_paths, train_mask_paths)
print(len(train_dataset[0]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)

valid_dataset = CustomDataset(valid_image_paths, valid_mask_paths)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True, num_workers=2)

test_dataset = CustomDataset(test_image_paths, test_mask_paths)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)
  


dataLoaders = {
        'train': train_loader,
        'valid': valid_loader,
        }


