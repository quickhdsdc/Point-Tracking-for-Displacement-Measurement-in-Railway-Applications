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
def train(train_loader, valid_loader, model, criterion, optimizer, 
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
    train_loss_min = np.Inf
    
    train_losses = []
    valid_losses = []
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [90, 110], 0.1)
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
            # perform a single optimization step (parameter update)
            optimizer.step()
            # lr_scheduler.step()
            # print(lr_scheduler.get_last_lr())
            # update running training loss
            train_loss += loss.item()*batch['image'].size(0)
            count_train += 1
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
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
                  .format(valid_loss_min, valid_loss))
            torch.save(model.state_dict(), saved_model)
            valid_loss_min = valid_loss
            
        # if train_loss <= train_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'
        #           .format(train_loss_min, train_loss))
        #     torch.save(model.state_dict(), saved_model)
        #     train_loss_min = train_loss            
    return train_losses, valid_losses 