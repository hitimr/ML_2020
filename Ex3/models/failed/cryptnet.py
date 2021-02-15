# System Constants
MNIST_IMG_HEIGHT = 28   # image height
MNIST_IMG_HWIDTH = 28   # image width
MNIST_PIXEL_CNT = MNIST_IMG_HEIGHT * MNIST_IMG_HWIDTH
NUM_CLASSES = 10

import crypten.nn as crpyt_nn

# Architecture
# Depth structure of the model aka. number and size of the layers
# this is function is called in the Net.__init__()
def layers_crypten(self):
        self.fc1 = crpyt_nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc2 = crpyt_nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc3 = crpyt_nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc4 = crpyt_nn.Linear(MNIST_PIXEL_CNT, MNIST_PIXEL_CNT)
        self.fc5 = crpyt_nn.Linear(MNIST_PIXEL_CNT, 10)

# forward function used for the net
def forward_crypten(self, x): 
        # flatten input     
        x = x.view(-1, MNIST_PIXEL_CNT)
               
        # Pytorch does not support a (binary) sign activation fucntion
        # so we had to create our own actctivator
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
    

class CryptNet(crpyt_nn.Module):
    def __init__(self):
        super(CryptNet,self).__init__()    
        layers_crypten(self) # defined in the loaded conf file

    def forward(self, x):   
        return forward_crypten(self, x) # defined in loaded conf file
    
test = CryptNet()