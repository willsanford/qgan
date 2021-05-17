from pennylane.ops.qubit import PauliZ
import torch
from torch import nn 
import numpy as np
import pennylane as qml

# This is a simple class that will test a quantum generator agaist quantum discriminator. The style of this architecture is similar
# to the architeture defined in Pennylane's QGan tutorial. The data we are trying to 


# Generics that will be used in this model
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
eps = 1e-2 

class MnistClassical:
    def __init__(self, logger, device):
        self.l = logger
        self.device = device

        # Image constants
        image_size = 784
        hidden_size = 256
        latent_size = 64

        # Set discriminator architecture
        self.D = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())
        self.D.to(self.device)

        # Set the generator architecture
        self.G = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())      
        self.G.to(device)


        
    def step(self):
        pass
        # return disc_loss.item(), gen_loss.item()

    def save_checkpoint(self, model, path):
        try:
            if model == 'gen':
                torch.save(self.g_weights, path)
            else:
                torch.save(self.d_weights, path)
        except:
            return False
        else:
            return True
        