import torchvision.transforms as transforms
import torch.nn as nn

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from config import *


# System Constants
MNIST_IMG_HEIGHT = 28
MNIST_IMG_HWIDTH = 28
MNIST_PIXEL_CNT = MNIST_IMG_HEIGHT * MNIST_IMG_HWIDTH

# --- Model Architecture
ARCHITECTURE = "binary_mnist" # MPyC Architecture

def architecture(self):
        self.fc1 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc2 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc3 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc4 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc5 = nn.Linear(MNIST_PIXEL_CNT, 10)
        

# --- Data Loader
batch_size = 20  # how many samples per batch to load
valid_size = 0.2 # percentage of training set to use as validation
num_workers = 0 # number of subprocesses to use for data loading

# --- Data preparation
# Thresholding
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x > 0,
        lambda x: x.float(),
])

# Model Settings
model_file_name = MODEL_DIR + "mnist_binary.pt"

# Train Settings
n_epochs = 5 # number of epochs to train the model

