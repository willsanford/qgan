# This file houses all the functions that are needed by multiple files


# Common function header

# def funct(a: type, b: type, c: type) -> type:
#     '''
#       Short description of the function

#       Args:
#         a: short desc
#         b: short desc
#         c: short desc

#       Returns:
#         return value : short desc
#     '''

from datetime import date, datetime
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch

def load_meta(file):
    '''
    Load the information from the file 'meta.txt' and return a dictionry with the corresponding metadata and its values

    Args: 
        None
    Returns:
        Dict[str,str]: a dictionary of the meta parameters
    '''
    f = open(file, 'r')
    out = {}
    for line in f.readlines():
        # Ignore comments and blank lines
        if line[0] != '#' and len(line) != 1:
            s = line.strip().split(' ')
            out[s[0]] = s[1]
    f.close()
    return out

def get_model(name: str, logger=None, device=None):
    '''
    Initialize a given model from a string without having to load each on into a dictionary
    Args: 
        name: The name of the model to initialize and return
    Returns:
        Model(): return an initialized model
    '''
    if name == 'basic_q':
        from models.basic_q import InitialQuantumModel
        return InitialQuantumModel()
    elif name == 'generic_g':
        from models.test_models import Generator
        return Generator()
    elif name == 'generic_d':
        from models.test_models import Discriminator
        return Discriminator()
    elif name == 'mnist_classical':
        from models.mnist_classical import MnistClassical
        return MnistClassical(logger,device)

    
def visualize_losses(save_path: str,
                     losses = Tuple):
    '''
    Takes a tuple of losses (gen and disc) and saves matpolotlib image

    Args: 
        save_path: path to save visualizations to 
        losses: tuple of lists of gen and disc losses
    Returns:
        None
    '''
    # Unpack tuple
    g_losses, d_losses = losses
    x_values = range(len(g_losses))
    plt.plot(x_values, g_losses, label = 'gen')
    plt.plot(x_values, d_losses, label = 'disc')
    plt.legend()
    plt.savefig(save_path)
    plt.show()


def get_cuda():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')