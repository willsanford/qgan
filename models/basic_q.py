import torch
from torch import nn 
import pennylane as qml
from generic import Discriminator, Generator
from typing import Dict
# This is a simple class that will test a quantum generator agaist a classical. The style of this architecture is similar
# to the architeture defined in Pennylane's QGan tutorial



class qDisc(Discriminator):
    def __init__(self):
        super().__init__('Basic Quantum Disciminator')

    def circuit(in):
        pass




class qGen(Generator):
    def __init__(self):
        
        super().__init__('Basic Quantum Generator')     
        
    dev = qml.device('default.qubit', wires = 3)
    @qml.qnode(dev, interface='torch')
    def forward_pass(self, inputs: Dict[str, torch.tensor]) -> Dict[str, torch.tensor]:

        weights = inputs['weights']

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

        return {
            'gen_out': qml.expval(qml.PauliZ(2))
        }

        

