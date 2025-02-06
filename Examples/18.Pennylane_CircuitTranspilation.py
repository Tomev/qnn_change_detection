# For our noisy training to work, we have to have the Pennylane circuit comiled for
# actual noisy device. That's especially true for the basis gate set.

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn


def main():

    # Config
    n_qubits = 4
    dev = qml.device("default.qubit", wires=int(2 * n_qubits))

    def circuit(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(2 * n_qubits), rotation="Y")
        qml.SimplifiedTwoDesign(
            initial_layer_weights=[0] * n_qubits * 2,
            weights=weights,
            wires=range(int(2 * n_qubits)),
        )

        # return qml.expval(qml.PauliZ(1))

    shapes = qml.SimplifiedTwoDesign.shape(n_layers=3, n_wires=int(2 * n_qubits))[1]

    w = np.random.random(size=shapes)

    # print(w)

    x = [np.random.random() for _ in range(n_qubits * 2)]

    compiled_circuit = qml.compile(
        circuit, basis_set=["SX", "X", "RZ", "ECR"], num_passes=2, expand_depth=10
    )

    drawer = qml.draw(compiled_circuit)
    print(drawer(x, w))

    print("\tCompilation, unfortunately, fails.")
    print(
        "\tSee: https://discuss.pennylane.ai/t/how-to-transpile-or-compile-a-circuit-with-a-specific-hardware-setting/3047"
    )


if __name__ == "__main__":
    print("Start!")
    main()
    print("Done!")
