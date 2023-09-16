from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
from functions import *
from train import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Define path
    traindir = "./data/MNIST/"
    validdir = "./data/MNIST/"

    # Change to fit hardware
    num_workers = 0

    batch_size = 64
    n_epochs = 5

    learning_rate = 0.001


    max_epochs_stop = 3
    print_every = 1

    device = torch.device("cpu")
    print('Device: {} Epochs: {} Batch size: {}'.format(device, n_epochs, batch_size))
    # define image transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data= datasets.MNIST(traindir, train=True, download=True, transform=transform)
    valid_data = datasets.MNIST(validdir, train=False, download=True, transform=transform)

    print('Length train: {} Length valid: {}'.format(len(train_data), len(valid_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,shuffle=True,  num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size,shuffle=False,  num_workers=num_workers)
    print('Number of train batches: {} Number of valid batches: {}'.format(len(train_loader), len(valid_loader)))

    categories = []
    for d in os.listdir(traindir):
        categories.append(d)

    n_classes = len(categories)

    inputs, classes = next(iter(train_loader))
    out = torchvision.utils.make_grid(inputs)
    imshow_tensor(out, title=[train_data.classes[x] for x in classes])


    model = mobilenetv3_small()
    model.num_class = n_classes
    #"""
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

    #cover class to index
    model.class_to_idx = train_data.class_to_idx
    model.idx_to_class = {
        idx: class_
        for class_, idx in model.class_to_idx.items()
    }
    #"""
    # Set up your loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), learning_rate)

    save_file_name = 'mobilenet_v3_model_best_model.pt'

    model, history = train(model,
                           criterion,
                           optimizer,
                           train_loader,
                           valid_loader,
                           save_file_name,
                           max_epochs_stop,
                           n_epochs,
                           print_every)
    """
    dataiter = iter(iter(train_loader))
    # get some random training images
    # you may use .next() to get the next iteration of validation dataloader

    images, labels = dataiter.__next__()
    images = images.repeat(1, 3, 1, 1)
    outputs = model(images.float())
    _, predicted = torch.max(outputs, 1)
    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' % categories[labels[j].long()] for j in range(batch_size)))
    print('Prediction: ', ' '.join('%5s' % categories[predicted[j].long()] for j in range(batch_size)))
    """

    # Final prediction

    ids = list(range(batch_size))
    result = pd.DataFrame(ids, columns=['id'])
    predictions = []
    real = []
    validationiter = iter(valid_loader)
    dataiter = iter(validationiter)

    data, target = dataiter.__next__()
    data = data.repeat(1, 3, 1, 1)

    outputs = model(data.float())
    pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    predictions += list(pred.cpu().numpy()[:, 0])
    real += list(target.numpy())
    result['pred'] = predictions
    result['real'] = real
    result.to_csv('result.csv', index=False)
    print('Result saved in: {}'.format('result.csv'))