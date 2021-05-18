import torch
from torch import nn
from torch._C import device 
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.datasets import MNIST
import numpy as np

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
        
        self.batch_size = 10

        # Download the MNIST dataset
        mnist = MNIST(root='data', 
              train=True, 
              download=True,
              transform=Compose([ToTensor(), Normalize(mean=(0.5,), std=(0.5,))]))
        
        self.data_loader = DataLoader(mnist, self.batch_size, shuffle=True)
        
        self.l.log('LOG', 'MNIST loaded.')

        # Image constants
        self.image_size = 784
        self.hidden_size = 256
        self.latent_size = 64    

        # Set discriminator architecture
        self.d = nn.Sequential(
            nn.Linear(self.image_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid())
        self.d.to(self.device)

        self.d_optim = Adam(self.d.parameters(), lr=0.0002)

        # Set the generator architecture
        self.g = nn.Sequential(
            nn.Linear(self.latent_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.image_size),
            nn.Tanh())      
        self.g.to(device)
        self.g_optim = Adam(self.g.parameters(), lr=0.0002)

        self.criterion = nn.BCELoss()

    def step_disc(self, images):
        print(images)
        real_labels = torch.ones(self.batch_size, 1).to(self.device)
        fake_labels = torch.zeros(self.batch_size, 1).to(self.device)

        # Discriminate on real images and calcualte loss
        out = self.d(images)    
        real_loss = self.criterion(out, real_labels)

        # Discriminate on fake images and calculate loss
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        fake_in = self.g(z)
        fake_out = self.d(fake_in)
        fake_loss = self.criterion(fake_out, fake_labels)

        total_loss = fake_loss + real_loss

        # Reset gradient and calculate backprop
        self.d_optim.zero_grad()
        total_loss.backward()
        self.d_optim.step()

        return total_loss

    def step_gen(self):
        # Generate random images
        z = torch.randn(self.batch_size, self.latent_size).to(self.device)
        fake_in = self.g(z)
        labels = torch.ones(self.batch_size, 1).to(self.device)

        # Calculate loss and gradients
        loss = self.criterion(self.d(fake_in), labels)
        self.g_optim.zero_grad()
        loss.backward()
        self.g_optim.step()

        return loss

    def step(self):
        images, _ = next(iter(self.data_loader))
        d = self.step_disc(images)
        g = self.step_disc()
        return d, g

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
        