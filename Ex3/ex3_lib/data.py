"""data.py
Helper functions for splitting the dataset
"""
import numpy as np

def get_indices(length, num_splits):
    step = int(length / num_splits)
    liste = np.arange(0, length+1, step=step, dtype=int, )
    if liste[-1]!=length:
        liste[-1]=length
    return liste
    
def split_data(data, frac):
    length = len(data[1]) #.shape[0]
    split_idx = int(length*frac)
    print(f"Returning: 0 <-1-> {split_idx} <-2->{length}")
    feats_1, labels_1 = data[0][:split_idx], data[1][:split_idx]
    feats_2, labels_2 = data[0][split_idx:], data[1][split_idx:]
    return (feats_1, labels_1), (feats_2, labels_2)

def split_data_even_old(data, num_splits:int, length=None):
    if length==None:
        length = len(data[1]) #.shape[0]
    split_idx = get_indices(length, num_splits)
    return [[data[0][start:stop], data[1][start:stop]] for (start, stop) in zip(split_idx[:-1], split_idx[1:])]

def split_data_even(data, num_splits:int, length=None):
    if length==None:
        length = len(data[1]) #.shape[0]
    split_idx = get_indices(length, num_splits)
    return [data[start:stop] for (start, stop) in zip(split_idx[:-1], split_idx[1:])]

if __name__=="__main__":
    pass
