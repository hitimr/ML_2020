import torch

# Pytorch does not support a (binary) sign activation fucntion
# therefore we need to define our own
# TODO: use binary sign
def sgn(input):
    output = torch.sign(input)
    return output