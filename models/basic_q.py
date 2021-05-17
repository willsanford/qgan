import torch
import numpy as np
import pennylane as qml

# This is a simple class that will test a quantum generator agaist quantum discriminator. The style of this architecture is similar
# to the architeture defined in Pennylane's QGan tutorial. The data we are trying to 


# Generics that will be used in this model
phi = np.pi / 6
theta = np.pi / 2
omega = np.pi / 7
eps = 1e-2 

class InitialQuantumModel:
    def __init__(self):
        # Set generator architecture
        self.g_weights = torch.tensor(np.array([np.pi] + [0] * 8) + np.random.normal(scale=eps, size=(9,)), requires_grad=True)
        self.g_optimizer = torch.optim.Adam([self.g_weights], lr=0.1)

        self.d_weights = torch.tensor(np.random.normal(size=(9,)), requires_grad=True)
        self.d_optimizer = torch.optim.Adam([self.d_weights], lr=0.1)

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

        def gen(weights):
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
        
        def discrim(weights):
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
        disc_cost = lambda disc_weights: real_given_fake(self.g_weights, disc_weights) - real_given_real(disc_weights)
        gen_cost = lambda gen_weights: -real_given_fake(gen_weights, self.d_weights)

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
        