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

#
# ---- Dataset
#
# https://github.com/rois-codh/kmnist
DATASET_FULLNAME = "Fashion MNIST"
DATASET_NAME = "fashion"
DATASET_LINK = "https://github.com/zalandoresearch/fashion-mnist"
DATASET_DESCRIPTION = "Fashion-MNIST is a dataset of Zalando's article images—consisting of a training set of 60,000 examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10 classes. We intend Fashion-MNIST to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms. It shares the same image size and structure of training and testing splits."
# System Constants
#
IMG_HEIGHT = 28   # image height
IMG_WIDTH = 28   # image width
PIXEL_CNT = IMG_HEIGHT * IMG_WIDTH
NUM_CHANNELS = 1 # grayscale
NUM_CLASSES = 10
SIZE_TRAIN = 60000
SIZE_TEST = 10000
SIZE = SIZE_TRAIN + SIZE_TEST

#
# ---- Model 
#
# File name and path for the final model
bmlp_file_name = MODEL_DIR + f"{DATASET_NAME}_binary.pt"   
model_file_name = bmlp_file_name
#
# --- Architectures
#
### BMLP
# Binarized multi-layer perceptron
# Mirrors the implementation of the MPyC team in pytorch.
def layers(self):
    """
    Depth structure of the model aka. number and size of the layers.
    This is function is called in the Net.__init__()
    Intended as a friendly function for a pytorch model.
    
    CODE:
    def layers(self):
        self.fc1 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
        self.fc2 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
        self.fc3 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
        self.fc4 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
        self.fc5 = nn.Linear(PIXEL_CNT, 10)
    """
    self.fc1 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
    self.fc2 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
    self.fc3 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
    self.fc4 = nn.Linear(PIXEL_CNT, PIXEL_CNT)
    self.fc5 = nn.Linear(PIXEL_CNT, 10)

def forward(self, x): 
    """
    intended as a friendly function for a pytorch model.
    Execution flow of the model.
    This is function is called in the Net.forward().
    
    CODE:
    def forward_relu(self, x): 
        # flatten input     
        x = x.view(-1, MNIST_PIXEL_CNT)
        x = self.fc1(x)
        x = x.relu()
        x = self.fc2(x)
        x = x.relu()
        x = self.fc3(x)
        x = x.relu()
        x = self.fc4(x)
        x = x.relu()
        x = self.fc5(x)
        return x
    """
    # flatten input     
    x = x.view(-1, PIXEL_CNT)
    x = self.fc1(x)
    x = x.sign()
    x = self.fc2(x)
    x = x.sign()
    x = self.fc3(x)
    x = x.sign()
    x = self.fc4(x)
    x = x.sign()
    x = self.fc5(x)
    return x

class ReLUMLP(nn.Module):
    def __init__(self):
        super(ReLUMLP, self).__init__()    
        layers(self) # defined in the loaded conf file

    def forward(self, x):   
        return forward(self, x) # defined in loaded conf file


#
# --- Data
# Loader
sample_size = 0  # Number of images from the train set that are actually used for training. (0 = use all)
batch_size = 20  # how many samples per batch to load
valid_size = 0.2 # percentage of training set to use as validation
num_workers = 0 # number of subprocesses to use for data 
 
    
#
# --- Data preparation
# Thresholding
blackwhite_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        lambda x: x > 0,
        lambda x: x.float(),
])
# bound as the default transform for our scripts
transform = blackwhite_transform


#
# --- Training
#
n_epochs = 5 # number of epochs to train the model

