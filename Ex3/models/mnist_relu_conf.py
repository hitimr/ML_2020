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
DATASET_FULLNAME = "MNIST database of handwritten digits"
DATASET_NAME = "mnist"
DATASET_LINK = "http://yann.lecun.com/exdb/mnist/"
DATASET_DESCRIPTION = "MNIST database of handwritten digits."
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


# ---- Model 
#
# File name and path for the final model
relu_file_name = MODEL_DIR + f"/{DATASET_NAME}_relu.pt"   
model_file_name = relu_file_name
#
# --- Architectures
#
### RELU MLP
    
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
    x = x.relu()
    x = self.fc2(x)
    x = x.relu()
    x = self.fc3(x)
    x = x.relu()
    x = self.fc4(x)
    x = x.relu()
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
sample_size = int(SIZE_TRAIN*0.05)  # Number of images from the train set that are actually used for training. (0 = use all)
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
n_epochs = 2 # number of epochs to train the model
learning_rate = 0.001

