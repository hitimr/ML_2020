import torchvision.transforms as transforms
import torch.nn as nn

# macro for accessing parent folder
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

#custom modules
from config import *
from common import *


# System Constants
MNIST_IMG_HEIGHT = 28   # image height
MNIST_IMG_HWIDTH = 28   # image width
MNIST_PIXEL_CNT = MNIST_IMG_HEIGHT * MNIST_IMG_HWIDTH


   

# ---- Model 
# File name and path for the final model
model_file_name = MODEL_DIR + "mnist_binary.pt"   

# Architecture
# Depth structure of the model aka. number and size of the layers
# this is function is called in the Net.__init__()
def layers(self):
        self.fc1 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc2 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc3 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc4 = nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc5 = nn.Linear(MNIST_PIXEL_CNT, 10)

# forward function used for the net
def forward(self, x): 
        # flatten input     
        x = x.view(-1, MNIST_PIXEL_CNT)
               
        # Pytorch does not support a (binary) sign activation fucntion
        # so we had to create our own actctivator
        x = sgn(self.fc1(x))
        x = sgn(self.fc2(x))
        x = sgn(self.fc3(x))
        x = sgn(self.fc4(x))
        x = self.fc5(x)
        return x

# --- Data
# Loader
sample_size = 0  # Number of images from the train set that are actually used for training. (0 = use all)
batch_size = 20  # how many samples per batch to load
valid_size = 0.2 # percentage of training set to use as validation
num_workers = 0 # number of subprocesses to use for data 
 

# --- Data preparation
# Thresholding
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x > 0,
        lambda x: x.float(),
])


# --- Training
n_epochs = 5 # number of epochs to train the model

