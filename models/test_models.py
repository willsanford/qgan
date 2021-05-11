import torch
from generic import Discriminator, Generator
from typing import Dict
import numpy as np

# A simple test generator to see if other parts of the architecture work 
class TestGen(Generator):
    def __init__(self, out_size):
        super(TestGen, self).__init__(self, 'test generator')
        self.output_dim = out_size
        
    def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        return {
            'gen_out': torch.rand(self.output_dim)
        }

# A simple test discriminator to see if other parts of the architecture work 
class TestDisc(Discriminator):
    def __init__(self):
        super(TestDisc, self).__init__(self, 'test discriminator')
        
    def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:
        return {
            'disc_out': torch.rand((1))
        }