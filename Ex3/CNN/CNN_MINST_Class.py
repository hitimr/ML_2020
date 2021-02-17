from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

import CNN_model as CNN
import CNN_functions as functions

import time

# in das terminal eingeben
# python3 CNN_MINST_Class.py --test-batch-size 100 --epochs 2 --batch-size 128 --first_layer_in 32 --first_layer_out 64


if __name__ == '__main__':
        # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--first_layer_in', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 1000)')
    parser.add_argument('--first_layer_out', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=3, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                    help='For Saving the current Model')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = CNN.Net(args.first_layer_in,args.first_layer_out).to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    accuracies = []
    tottime = 0
    parameter_set = {}
    results = pd.DataFrame()
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        memory, inference_time = functions.train(args, model, device, train_loader, optimizer, epoch)
        #CNN.train_CNN(args, model, device, train_loader, optimizer, epoch)
        accur, y_pred_array, y_true_array = functions.test(model, device, test_loader)
        con_mat = confusion_matrix(y_true_array, y_pred_array, labels=np.arange(0,10))
        heat_map = sns.heatmap(con_mat, annot=True)
        plt.savefig("out/heat_map")
        accuracies.append(accur)
        scheduler.step()
        end = time.time()
        tottime += end - start
        parameter_set["epochtime"] = end - start
        parameter_set["accuracy"] = accur
        parameter_set["inference_time"] = inference_time
        parameter_set["efficience"] = accur/(end - start)
        parameter_set["total Memory"] = memory
        results = results.append(parameter_set, ignore_index=True)
        print("epochtime = {} s; efficience = {}\n".format((end - start),accur/(end - start)))
    #print("epochtime {} s".format((end - start)))
    #print("tottime {} s".format(tottime)))
    plt.plot(accuracies)
    plt.grid()
    plt.savefig("out/acc_plot")
    print("inference_time: ",inference_time,"s")
    print("total memory: ",memory,"Mb")
    results.to_csv("out/Hallo.csv", index=False)
    print(results)