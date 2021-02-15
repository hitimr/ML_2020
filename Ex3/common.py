import torch


# TODO: use binary sign
def sgn(input):
    """
    binary sign (activation) function
    
    Pytorch does not support a (binary) sign activation fucntion
    therefore we need to define our own.
    """
    output = torch.sign(input)
    return output


import matplotlib.pyplot as plt
import numpy as np

def plot_batch(images, labels, predictions=None, label_color="black"):
    """
    Plot the images in the batch, along with the corresponding labels.
    If predictions are given, will also show the prediction above the image
    in red if wrong or green if right. (invalidates arg @label_color)
    
    Parameters:
        - images
        - labels
        - predictions [optional]
        - label_color [optional]... color for labels above image
    """
    images = images.numpy()
    fig = plt.figure(figsize=(25, 4))
    for idx in np.arange(20):
        ax = fig.add_subplot(2, int(20/2), idx+1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        if predictions != None:
            ax.set_title("{} ({})".format(str(predictions[idx].item()), str(labels[idx].item())),
                         color=("green" if predictions[idx]==labels[idx] else "red"))
        else:
            ax.set_title(str(labels[idx].item()), color=label_color)