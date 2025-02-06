import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def main() -> None:

    n_qubits = 4
    n_wires = 2 * n_qubits

    dev = qml.device("default.qubit", wires=n_wires)

    @qml.qnode(dev)
    def qnode(inputs, weights):
        init_weights = [0, 0, 0, 0, 0, 0, 0, 0]
        qml.templates.AngleEmbedding(inputs, wires=range(2 * n_qubits), rotation="Y")
        qml.SimplifiedTwoDesign(
            initial_layer_weights=init_weights,
            weights=weights,
            wires=range(n_wires),
        )
        return qml.expval(qml.PauliZ(1))

    shape = qml.SimplifiedTwoDesign.shape(n_layers=3, n_wires=n_wires)[1]

    print(shape)
    weight_shapes = {"weights": shape}

    quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    rand_input = torch.rand(n_wires)
    # rand_weights = torch.rand(weight_shapes)

    print(quantum_layer.forward(rand_input))


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
