from pennylane.ops.qubit import PauliZ
import torch
from torch import nn 
import numpy as np
import pennylane as qml
from generic import Discriminator, Generator
from typing import Dict
# This is a simple class that will test a quantum generator agaist quantum discriminator. The style of this architecture is similar
# to the architeture defined in Pennylane's QGan tutorial. The data we are trying to 


# Generics that will be used in this model
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
eps = 1e-2 


# TODO: make the generic version of this system work like the single class version does
# class qDisc(Discriminator):
#     def __init__(self):
#         super().__init__('Basic Quantum Disciminator')

#     @qml.qnode(dev, interface='torch')
#     def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:

#         weights = inputs['disc_weights']

#         qml.Hadamard(wires=0)
#         qml.RX(weights[0], wires=0)
#         qml.RX(weights[1], wires=2)
#         qml.RY(weights[2], wires=0)
#         qml.RY(weights[3], wires=2)
#         qml.RZ(weights[4], wires=0)
#         qml.RZ(weights[5], wires=2)
#         qml.CNOT(wires=[0, 2])
#         qml.RX(weights[6], wires=2)
#         qml.RY(weights[7], wires=2)
#         qml.RZ(weights[8], wires=2)

#         return {
#             'disc_out': qml.expval(qml.PauliZ(2))
#         }


# class qGen(Generator):
#     def __init__(self):
#         super().__init__('Basic Quantum Generator')     
        
#     @qml.qnode(dev, interface='torch')
#     def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:

#         weights = inputs['gen_weights']

#         qml.Hadamard(wires=0)
#         qml.RX(weights[0], wires=0)
#         qml.RX(weights[1], wires=1)
#         qml.RY(weights[2], wires=0)
#         qml.RY(weights[3], wires=1)
#         qml.RZ(weights[4], wires=0)
#         qml.RZ(weights[5], wires=1)
#         qml.CNOT(wires=[0, 1])
#         qml.RX(weights[6], wires=0)
#         qml.RY(weights[7], wires=0)
#         qml.RZ(weights[8], wires=0)

#         return {}


#     def real_output(inputs: Dict[str, torch.tensor]):
#         angles = inputs['angles']

#         qml.Hadamard(wires=0)
#         qml.Rot(*angles, wires=0)

class InitialQuantumModel():
    def __init__(self, gen_optimizer, disc_optimizer):
        # Set generator architecture
        # self.gen = qDisc()
        self.g_optimizer = gen_optimizer
        self.g_weights = np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=(9,))

        # self.disc = qDisc()
        self.d_optimizer = disc_optimizer
        self.d_weights = np.random.normal(size=(9,))

    def step(self):
        # Define a device for PennyLane syntax
        dev = qml.device('default.qubit', wires = 3)

        # ##############################
        # Define training loop functions
        # ##############################
        
        # Functions must be defined in function to properly decorate with PennyLane Syntax
        def real(angles):
            qml.Hadamard(wires=0)
            qml.Rot(*angles, wires=0)

        def gen(self, weights):
            qml.Hadamard(wires=0)
            qml.RX(weights[0], wires=0)
            qml.RX(weights[1], wires=1)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=1)
            qml.RZ(weights[4], wires=0)
            qml.RZ(weights[5], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RX(weights[6], wires=0)
            qml.RY(weights[7], wires=0)
            qml.RZ(weights[8], wires=0)
        
        def discrim(self, weights):
            qml.Hadamard(wires=0)
            qml.RX(weights[0], wires=0)
            qml.RX(weights[1], wires=2)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=2)
            qml.RZ(weights[4], wires=0)
            qml.RZ(weights[5], wires=2)
            qml.CNOT(wires=[0, 2])
            qml.RX(weights[6], wires=2)
            qml.RY(weights[7], wires=2)
            qml.RZ(weights[8], wires=2)

        # Define a real data and generated data discriminator circuit
        @qml.qnode(dev, interface='torch')
        def disc_real(inputs, disc_weights):
            real(inputs)
            discrim(disc_weights)
            return qml.expval(qml.PauliZ(2))

        @qml.qnode(dev, interface='torch')
        def disc_gen(gen_weights, disc_weights):
            gen(gen_weights)
            discrim(disc_weights)
            return qml.expval(qml.PauliZ(2))

        # TODO: lots of lambda nesting here. Add more explanation or just make real functions
        # Define probability functions
        real_given_real = lambda disc_weights: (disc_real([phi, theta, omega], disc_weights) + 1) / 2
        real_given_fake = lambda gen_weights, disc_weights: (disc_gen(gen_weights, disc_weights) + 1) / 2

        # We only want one weight to be learned in each, so one will be a static .self variable
        disc_cost = lambda disc_weights: real_given_fake(self.gen_weights, disc_weights) - real_given_real(disc_weights)
        gen_cost = lambda gen_weights: -real_given_fake(gen_weights, self.disc_weights)

        # Train the generator
        self.g_optimizer.zero_grad()
        gen_loss = gen_cost(self.g_weights)
        gen_loss.backward()
        self.g_optimizer.step()

        # Train the optimizer
        self.d_optimizer.zero_grad()
        disc_loss = disc_cost(self.d_weights)
        disc_loss.backward()
        self.d_optimizer.step()

        return disc_loss.item(), gen_loss.item()