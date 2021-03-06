import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import numpy as np
import psutil
import time

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    memory = []
    start = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        memory_start = psutil.virtual_memory().total - psutil.virtual_memory().available
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            memory_end = psutil.virtual_memory().total - psutil.virtual_memory().available
            memory_diff = np.abs(memory_end - memory_start)/(1024**2)
            memory.append(memory_diff)
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
    end = time.time()
    return np.max(memory), end - start


def test(model, device, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            y_pred.extend(list(pred.numpy()))
            y_true.extend(list(target.numpy()))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset), np.array(y_pred), np.array(y_true)
