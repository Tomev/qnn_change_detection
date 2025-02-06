import os
from typing import Dict

# import pennylane as qml
import qiskit
import torch
import torch.nn as nn
from iqm.qiskit_iqm import IQMProvider
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from pennylane import numpy as np
# from qiskit_aer.noise import NoiseModel
# import qiskit_aer.noise as noise
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import PauliTwoDesign
from qiskit.providers.backend import BackendV1, BackendV2
from qiskit.qasm3 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN, SamplerQNN


def layout_to_map(layout, quantum_register):
    # Get simple qubit map from circuit layout.
    # The map is a dict in form {virtual -> physical}.
    # We want to act on the physical qubits.
    initial_layout = layout.initial_layout.get_virtual_bits()
    # print(test)

    # print(initial_layout)

    qubit_map = {}

    for k in quantum_register:
        qubit_map[k._index] = initial_layout[k]

    return qubit_map


def SimplifiedTwoDesignQiskit(n_qubits, n_layers) -> QuantumCircuit:
    # Intended to be a qiskit version of the pennylane.SimplifiedTwoDesign circuit.
    # Shape of the weights should be (L, N-1, 2), where L is the number of layers and
    # N the number of qubits. Initial weights should have shape (N), if specified.

    # This version recreates the behavior of `compute_decomposition` method of the
    # `pennylane.SimplifiedTwoDesign` class. The only difference is that wires are given
    # by the number of qubits and indexed (0, ..., N - 1), because I'm not sure if
    # `qiskit.QuantumRegister` handles arbitrary qubit indices. The idea is to transpile
    # this circuit anyway, so that's not a big issue.

    # We want to obtain a parametrized circuit, so we made further modification
    weights = []
    for i in range(n_layers):
        weights.append(
            (
                ParameterVector(f"w_{i}_even", n_qubits - 1),
                ParameterVector(f"w_{i}_odd", n_qubits - 1),
            )
        )

    wires = range(n_qubits)

    c = QuantumCircuit(n_qubits)

    # SimplifiedTwoDesignLayers
    even_entanglers = [wires[i : i + 2] for i in range(0, len(wires) - 1, 2)]
    odd_entanglers = [wires[i : i + 2] for i in range(1, len(wires) - 1, 2)]

    for layer_weights in weights:
        # layer_weights should have shape (N-1, 2)

        # Even-part of the layer
        for q_pair in even_entanglers:
            c.cz(q_pair[0], q_pair[1])

        for q in range(n_qubits - 1):
            c.ry(layer_weights[0][q], q)

        # Odd-part of the layer
        for q_pair in odd_entanglers:
            c.cz(q_pair[0], q_pair[1])

        for q in range(n_qubits - 1):
            c.ry(layer_weights[1][q], q + 1)

    return c


def main() -> None:
    # get backend
    iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet"
    provider = IQMProvider(iqm_server_url)  # , token=token)
    backend = provider.get_backend()

    n_wires = 8

    pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
    # TODO TR: I assume `reps=3`, because that's the case in the quantum_classifier.py.
    ansatz = SimplifiedTwoDesignQiskit(n_qubits=n_wires, n_layers=3)
    transpiled_ansatz = pm.run(ansatz)

    print("\tTranspile feature map for selected backend.")
    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    # print("\tDefine qiskit.aer device with appropriate wires.")
    # self.dev = qml.device("qiskit.aer", wires=qubit_map.values(), shots=8000)  # Works
    # self.dev = qml.device("qiskit.remote", wires=qubit_map.values(), backend=AerSimulator())  # Has compilation errors!

    # print("\tDefine qiskit.remote device with appropriate wires.")
    # self.dev = qml.device("qiskit.remote", wires=backend.num_qubits, backend=backend)  #  WORKS!

    feature_map = QuantumCircuit(backend.num_qubits)

    data = ParameterVector("inputs", n_wires)  # Name "inputs" is important!

    for i, k in enumerate(qubit_map.keys()):
        feature_map.ry(data[i], qubit_map[k])

    pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
    transpiled_feature_map = pm.run(feature_map)

    # Network creation
    qnn_circuit = QNNCircuit(
        feature_map=transpiled_feature_map, ansatz=transpiled_ansatz
    )

    # print(circ.num_qubits)

    # create a simulator for backend
    sim = AerSimulator().from_backend(backend)

    estimator = Estimator(sim)
    gradient = ParamShiftEstimatorGradient(estimator, pass_manager=pm)

    observable = SparsePauliOp.from_sparse_list(
        [("Z", [qubit_map[1]], 1)], num_qubits=transpiled_ansatz.num_qubits
    )

    estimator_qnn = EstimatorQNN(
        circuit=qnn_circuit,
        observables=observable,
        estimator=estimator,
        gradient=gradient,
    )

    data = np.array([i / 2.0 for i in range(n_wires)], requires_grad=False)
    weights = np.array(
        [i / 10 for i in range(ansatz.num_parameters)], requires_grad=True
    )

    print(estimator_qnn.forward(data, weights))
    print(estimator_qnn.backward(data, weights))
    print(qnn_circuit)


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
