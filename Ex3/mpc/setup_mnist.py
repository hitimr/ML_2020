import torch
import numpy as np
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler
from config import *

from common import plot_batch

from models.mnist_relu_conf import *

def setup_data():
    # choose the training and testing datasets
    train_data = datasets.MNIST(root = "data", train = True, download = True, transform = transform)
    test_data = datasets.MNIST(root = "data", train = False, download = True, transform = transform)

    assert sample_size >= 0  # Error: Invalid TRAIN_SIZE
    assert sample_size <= len(train_data) # Error: Invalid TRAIN_SIZE

    # set number of subsamples 
    if sample_size == 0: num_train = len(train_data)
    else: num_train = sample_size

    # obtain training indices that will be used for validation 
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_index, valid_index = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)

    # prepare data loaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = train_sampler, num_workers = num_workers)
    valid_loader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, sampler = valid_sampler, num_workers = num_workers)
    test_loader =  torch.utils.data.DataLoader(test_data,  batch_size = batch_size, num_workers = num_workers)
