import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import torch.nn.functional as F
from torch import optim
import torch.nn as nn

device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## train
def train(train_loader, valid_loader, model, criterion, optimizer, lr_schedule=False,
          n_epochs=50, saved_model='model.pt'):
    '''
    Train the model
    
    Args:
        train_loader (DataLoader): DataLoader for train Dataset
        valid_loader (DataLoader): DataLoader for valid Dataset
        model (nn.Module): model to be trained on
        criterion (torch.nn): loss funtion
        optimizer (torch.optim): optimization algorithms
        n_epochs (int): number of epochs to train the model
        saved_model (str): file path for saving model
    
    Return:
        tuple of train_losses, valid_losses
    '''

    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf # set initial "min" to infinity   
    train_losses = []
    valid_losses = []
    if lr_schedule == 1:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 90, 120], 0.1)
    elif lr_schedule == 2:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 30, eta_min=0, last_epoch=-1)
    for epoch in range(n_epochs):
               
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0
        count_train=0
        count_val=0
        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch in train_loader:
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch['image'].to(device))
            # calculate the loss
            loss = criterion(output, batch['keypoints'].to(device))
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*batch['image'].size(0)
            count_train += 1
        # perform a single optimization step (parameter update)       
        if lr_schedule != 0:
            lr_scheduler.step()
        # print(lr_scheduler.get_last_lr())
        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        with torch.no_grad():
            for batch in valid_loader:
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(batch['image'].to(device))
                # calculate the loss
                loss = criterion(output, batch['keypoints'].to(device))
                # update running validation loss 
                valid_loss += loss.item()*batch['image'].size(0)
                count_val += 1

        # print training/validation statistics 
        # calculate average Root Mean Square loss over an epoch
        train_loss = np.sqrt(train_loss/count_train)
        valid_loss = np.sqrt(valid_loss/count_val)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'
              .format(epoch+1, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min and epoch>30:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            savepath = saved_model + '/epoch_' + str(epoch) + '.pt'
            torch.save(model.state_dict(), savepath)
            valid_loss_min = valid_loss
                        
    return train_losses, valid_losses 