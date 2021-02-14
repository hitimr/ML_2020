import torch

# Custom Activation functions
# TODO: use binary sign
def sgn(input):
    output = torch.sign(input)
    return output