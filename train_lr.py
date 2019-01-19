# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 18:14:28 2019

@author: Neda
"""

from split_dataset import dataLoaders
from U_Net_demo import model, device
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import time
import visdom

print("starting training")

#python -m visdom.server

def colorbar(mappable):
        ax = mappable.axes
        fig = ax.figure
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        return fig.colorbar(mappable, cax=cax)

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # 1e-3

# Decay LR by a factor of 0.1 every 20 epochs.
#step size: Period of learning rate decay.
#gamma = Multiplicative factor of learning rate decay. Default: 0.1, should float
scheduler = lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)


def train():
    
    
    since = time.time()
    
    vis = visdom.Visdom()    
    loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='Loss',
                                     title='Training Loss',
                                     legend=['Loss']))    
    vis = visdom.Visdom()
    accuracy_window = vis.line(X=torch.zeros((1,)).cpu(),
                           Y=torch.zeros((1)).cpu(),
                           opts=dict(xlabel='epoch',
                                     ylabel='accuracy',
                                     title='Training accuracy',
                                     legend=['accuracy']))          
    num_epochs=150
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))

        scheduler.step()
                
        model.train()  # Set model to training mode
                  
        train_loss = 0.0
        total_train = 0
        correct_train = 0
        
        #iterate over data
        for t_image, mask, image_paths, target_paths in dataLoaders['train']: 
                
                # get the inputs
                t_image = t_image.to(device)
                mask = mask.to(device)
                                
                 # zeroes the gradient buffers of all parameters
                optimizer.zero_grad()
                
                # forward
                outputs = model(t_image) 
                _, predicted = torch.max(outputs.data, 1)
                loss = criterion(outputs, mask) # calculate the loss
            
                # backward + optimize only if in training phase
                loss.backward() # back propagation
                optimizer.step() # update gradients                        
            
                # accuracy
                train_loss += loss.item()
                total_train += mask.nelement()  # number of pixel in the batch
                correct_train += predicted.eq(mask.data).sum().item() # sum all precited pixel values
                
        train_epoch_loss = train_loss / len(dataLoaders['train'].dataset)
        train_epoch_acc = 100 * (correct_train / total_train)
        
        print ('|train loss: {:.4f}| train ACC: {:.4f}|'.format(train_epoch_loss, train_epoch_acc))  
        print('-' * 70)        
                      
        vis.line(
                    X=torch.ones((1, 1)).cpu()*epoch,
                    Y=torch.Tensor([train_epoch_loss]).unsqueeze(0).cpu(),
                    win=loss_window,
                    update='append')
            
        vis.line(
                    X=torch.ones((1, 1)).cpu()*epoch,
                    Y=torch.Tensor([train_epoch_acc]).unsqueeze(0).cpu(),
                    win=accuracy_window,
                    update='append')
               
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))  
        
    torch.save(model.state_dict(), 'train_valid.pth')   
    
                                                   
    #visualize probability
    outputs = model(t_image)
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(10,6))
    
    img1 = ax1.imshow(torch.exp(outputs[0,0,:,:]).detach().cpu(), cmap = 'jet')
    ax1.axis('off')
    ax1.set_title("prob for class 0-train image")
    colorbar(img1)
    
    img2 = ax2.imshow(torch.exp(outputs[0,1,:,:]).detach().cpu(), cmap = 'jet')
    ax2.axis('off')
    ax2.set_title("prob for class 1-train image")
    colorbar(img2)
    
    img3 = ax3.imshow(torch.argmax(outputs, 1).detach().cpu().squeeze(), cmap = 'jet')
    ax3.axis('off')
    ax3.set_title("predicted for train image")
    colorbar(img3)
    
    plt.axis('on')
    plt.tight_layout(pad=0.01, w_pad=0.002, h_pad=1)
    plt.show()
    plt.savefig('train_visu.png')
           
                      
if __name__=='__main__':
    train()
  
