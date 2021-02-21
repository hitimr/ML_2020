import matplotlib.pyplot as plt
import numpy as np

def plot_digit(digit, label=None):
    fig = plt.figure()
    
    if label==None:
        plt.imshow(digit[0][0], cmap='gray', interpolation='none')
        plt.title(f"Label: {digit[1]}")
    
    else:
        plt.imshow(digit[0], cmap='gray', interpolation='none')
        plt.title(f"Label: {label}")

def plot_batch(batch, targets, b_range=8, grid=(4,2)):
    (cols, rows) = grid
    step = 1
    if isinstance(b_range, tuple):
        if len(b_range) ==2:
            (start, stop) = b_range
        elif len(b_range) ==3:
            (start, stop, step) = b_range
        else:
            raise ValueError("b_range Should have len==2")
    else:
        start, stop = 0, b_range
    fig = plt.figure()
    num_to_plot = int(np.ceil((stop - start) / step))
    rows = int(np.ceil(num_to_plot / cols))
    print(f"num_to_plot: {num_to_plot}")
    print(f"rows, cols = {rows}, {cols}")
    for i in range(start, stop, step):
        subplot_num = int(1 + ((i - start)/step) % num_to_plot)
        #print(f"i: {i} --> subplot#: {subplot_num}")
        plt.subplot(rows, cols, subplot_num)
        plt.imshow(batch[i][0], cmap='gray', interpolation='none')
        plt.title(f"Label(#{i}): {targets[i]}")
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    fig