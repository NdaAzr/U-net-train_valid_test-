# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:26:44 2019

@author: Neda
"""

from split_dataset import dataLoaders
from U_Net_demo import model, device
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import time
import numpy
import visdom

def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

criterion = nn.NLLLoss()



def validation( ):
    
    model.load_state_dict(torch.load('train_valid.pth'))
    
    vis = visdom.Visdom()    
    val_loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     title='validating Loss',
                                     legend=['Loss']))    
    vis = visdom.Visdom()
    val_accuracy_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='accuracy',
                                     title='validating accuracy',
                                     legend=['accuracy']))      
    
    since = time.time()
    
    num_epochs=200
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
       
                
        model.eval()  # Set model to training mode
                  
        valid_loss = 0.0
        total_valid = 0
        correct_valid = 0
        
            #iterate over data
        for t_image, mask, image_paths, target_paths in dataLoaders['valid']: 
                
                # get the inputs
                t_image = t_image.to(device)
                mask = mask.to(device)
                                
                
                # forward
                output = model(t_image) 
                _, predicted = torch.max(output.data, 1)
                loss = criterion(output, mask) # calculate the loss
                       
                # accuracy
                valid_loss += loss.item()
                total_valid += mask.nelement()  # number of pixel in the batch
                correct_valid += predicted.eq(mask.data).sum().item() # sum all precited pixel values
                
        valid_epoch_loss = valid_loss / len(dataLoaders['valid'].dataset)
        valid_epoch_acc = (correct_valid / total_valid)
        
        print ('valid loss: {:.4f} valid ACC: {:.4f}'.format(valid_epoch_loss, valid_epoch_acc))                                                                  
        print('-' * 30)  
        
        vis.line(
                    X=torch.ones((1, 1)).cpu()*epoch,
                    Y=torch.Tensor([valid_epoch_loss]).unsqueeze(0).cpu(),
                    win=val_loss_window,
                    update='append')
            
        vis.line(
                    X=torch.ones((1, 1)).cpu()*epoch,
                    Y=torch.Tensor([valid_epoch_acc]).unsqueeze(0).cpu(),
                    win=val_accuracy_window,
                    update='append')
               
        
    time_elapsed = time.time() - since
    print('validation complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  
    
       
    #visualize probability
    output = model(t_image)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10,6))
    
    img1 = ax1.imshow(torch.exp(output[0,0,:,:]).detach().cpu(), cmap = 'jet')
    ax1.axis('off')
    ax1.set_title("prob for class 0-train image")
    colorbar(img1)
    
    img2 = ax2.imshow(torch.exp(output[0,1,:,:]).detach().cpu(), cmap = 'jet')
    ax2.axis('off')
    ax2.set_title("prob for class 1-train image")
    colorbar(img2)
    
    img3 = ax3.imshow(torch.argmax(output, 1).detach().cpu().squeeze(), cmap = 'jet')
    ax3.axis('off')
    ax3.set_title("predicted for train image")
    colorbar(img3)
    
    plt.axis('on')
    plt.tight_layout(pad=0.01, w_pad=0.002, h_pad=1)
    plt.show()
    plt.savefig('train_visu.png')
        
                      
if __name__=='__main__':
    validation()
  
