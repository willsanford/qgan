# Generic generator/discriminator architecture that we can use to train more specific models
# Each generic will have two forward functions. One will be a simple 'forward' function that matches pytorches syntax which will simply call the other which will use stronger type casting
import torch
from torch import nn 
import pennylane as qml
from typing import List, Dict

class Discriminator(nn.Module):
    def __init__(self, 
                 name: str):
        super(Discriminator, self).__init__()
        self.name = name

    def get_name(self):
        return self.name
    
    def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        raise NotImplementedError
    
    def forward(self, inputs):
        return self.forward_pass(inputs)
    

class Generator(nn.Module):
    def __init__(self,
                 name: str):
        super(Generator, self).__init__()
        self.name = name

    def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        raise NotImplementedError

    def forward(self, inputs):
        return self.forward_pass(inputs)

    def real_ouput(self):
        raise NotImplementedError