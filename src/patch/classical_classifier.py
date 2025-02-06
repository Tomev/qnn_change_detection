# Author: Manish Kumar Gupta
# Date: 17/05/2024
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""

import sys
from turtle import forward

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


class ClassicalNeuralNetwork(nn.Module):
    def __init__(self, channels, n_qubits) -> None:
        super(ClassicalNeuralNetwork, self).__init__()
        self.channels = channels
        self.quant_conv_layers = nn.ModuleList()
        for i in range(self.channels):
            self.quant_conv_layers.append(torch.nn.Linear(n_qubits, 1))
        # self.classical_linear_layer0 = torch.nn.Linear(self.channels*2, 2)
        self.classical_linear_layer = torch.nn.Linear(self.channels, 5)
        self.classical_linear_layer1 = torch.nn.Linear(5, 2)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        num_features = 2 * x1.shape[2] * x1.shape[3]

        X = []
        for i in range(self.channels):
            x = torch.dstack(
                (torch.flatten(x1[:, i, :, :], 1), torch.flatten(x2[:, i, :, :], 1))
            )
            x = torch.reshape(x, (batch_size, num_features)).to("cuda")
            # print(x)
            x = self.quant_conv_layers[i](x).to("cuda")
            X.append(x)
            # print(x)
        # print(X)
        X = torch.dstack(X)
        # print(X)
        # X = torch.flatten(X,start_dim=1)
        # print(X)
        X = torch.squeeze(X)
        # print(X)

        # X = self.classical_linear_layer0(X)

        X = self.classical_linear_layer(X)
        X = self.classical_linear_layer1(X)

        output = self.sm(X)

        return output
