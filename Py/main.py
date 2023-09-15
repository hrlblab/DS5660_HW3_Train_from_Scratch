from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pandas as pd
from functions import *
from train import *



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define path
    traindir = "../data/MNIST/train"
    validdir = "../data/MNIST/val"
    #traindir = "data/MNIST/train"
    #validdir = "data/MNIST/val"
    # Change to fit hardware
    num_workers = 0
    batch_size = 64

    # define transforms
    # Image transformations
    """
    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
        # Validation does not use augmentation
        'valid':
            transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
    }

    # Datasets from folders

    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'valid':
            datasets.ImageFolder(root=validdir, transform=image_transforms['valid'])
    }

    # Dataloader iterators, make sure to shuffle
    dataloaders = {
        'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
        'val': torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    }
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dataset1 = datasets.MNIST('../data/MNIST/', train=True, download=False, transform=transform)
    dataset2 = datasets.MNIST('../data/MNIST/', train=False, transform=transform)

    #dataset1 = datasets.ImageFolder(root=traindir, transform=transform)
    #dataset2 = datasets.ImageFolder(root=validdir, transform=transform)
    print('Length train: {} Length test: {}'.format(len(dataset1), len(dataset2)))
    #dataloaders = {
    #    'train': torch.utils.data.DataLoader(data['train'], batch_size=batch_size, shuffle=True, num_workers=num_workers),
    #    'val': torch.utils.data.DataLoader(data['valid'], batch_size=batch_size, shuffle=True, num_workers=num_workers)
    #}
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=batch_size,shuffle=True,  num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset2, batch_size=batch_size,shuffle=False,  num_workers=num_workers)
    print('Number of train batches: {} Number of test batches: {}'.format(len(train_loader), len(valid_loader)))
    # Iterate through the dataloader once
    #trainiter = iter(dataloaders['train'])
    #validationiter = iter(dataloaders['val'])

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)

    n_classes = len(categories)
    #inputs, classes = next(iter(dataloaders['train']))
    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    # imshow_tensor(out, title=[data['train'].classes[x] for x in classes])


    model = mobilenetv3_small()
    model.num_class = n_classes
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #model.class_to_idx = data['train'].class_to_idx
    #"""
    model.class_to_idx = dataset1.class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }

    #list(model.idx_to_class.items())
    #"""
    # Set up your criterion and optimizer
    # You can use nn.CrossEntropyLoss() as your critenrion
    # You can use optim.Adam() as your optimizer with reasonable momentum
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for p in optimizer.param_groups[0]['params']:
        if p.requires_grad:
            print(p.shape)

    save_file_name = 'mobilenet_v3_model_best_model.pt'
    #"""
    model, history = train(model,
                           criterion,
                           optimizer,
                           #dataloaders['train'],
                           #dataloaders['val'],
                           train_loader,
                           valid_loader,
                           save_file_name=save_file_name,
                           max_epochs_stop=3,
                           n_epochs=10,
                           print_every=1)
    #"""
