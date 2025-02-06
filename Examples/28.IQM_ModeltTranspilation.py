# Let's see if I can create Pennylane circuit from transpiled qiskit circuit.

from typing import Dict

import pennylane as qml
import torch
import torch.nn as nn
from iqm.qiskit_iqm.fake_backends.fake_apollo import IQMFakeApollo
from pennylane import numpy as np
from pennylane_qiskit import load_noise_model
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliTwoDesign
from qiskit.qasm3 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import Estimator, QiskitRuntimeService
from qiskit_ibm_runtime.fake_provider import (FakeBrisbane, FakeMelbourne,
                                              FakeMelbourneV2)
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_qasm3_import import parse
from torch import Tensor


def layout_to_map(layout, quantum_register) -> Dict[int, int]:
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


def main() -> None:

    # https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
    # Example implementation.

    # Config
    algorithm_globals.random_seed = 42
    n_qubits = 4
    n_wires = 2 * n_qubits

    # Ansatz preparation

    # We want to start with ansatz, because it's more complicated than our selected
    # feature map (we use simple AngleEmbedding, which is a layer of single qubit
    # rotations). This way, we can transpile it, get the final layout of our circuit
    # and prepare the feature map and observable according to the layout obtained by
    # ansatz transpilation.

    ansatz = PauliTwoDesign(n_wires, reps=3)

    # Ansatz transpilation

    # backend = provider.get_backend('ibm_brisbane')
    simulator = IQMFakeApollo()
    # simulator = FakeBrisbane()

    # print(simulator.configuration().supported_instructions)

    # return

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    transpiled_ansatz = pm.run(ansatz)

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(f"\tPermutation map: {qubit_map}")

    # transpiled_ansatz_qasm = dumps(transpiled_ansatz)
    # transpiled_ansatz = parse(transpiled_ansatz_qasm)

    # print(transpiled_ansatz)

    # Feature map creation
    print(f"\tNumber of qubit in transpiled circuits: {transpiled_ansatz.num_qubits}")
    feature_map = QuantumCircuit(simulator.num_qubits)

    data = ParameterVector("inputs", n_wires)

    for i, k in enumerate(qubit_map.keys()):
        feature_map.ry(data[i], qubit_map[k])

    # Transpilation is needed, to ensure proper gates are used.
    # Ensure optimization level is low, so that the qubits are not permuted.
    pm = generate_preset_pass_manager(optimization_level=0, backend=simulator)
    transpiled_feature_map = pm.run(feature_map)

    print(transpiled_feature_map)
    print(transpiled_ansatz)

    # Pennylane transpilaiton
    # There's bug when compiling ECR gates via pennylane_qiskit.converter.
    # TODO: Analyze and submit it later. It can be easily fixed.
    # dev = qml.device("default.mixed", wires=qubit_map.values())  # Notice default.mixed
    dev = qml.device("default.qubit")

    # @qml.qnode(dev)
    def pennylane_circuit(inputs, weights):

        x = inputs.numpy()

        # w = weights.numpy()

        # qml.from_qiskit(transpiled_feature_map)(inputs)
        qml.from_qiskit(transpiled_feature_map)(x)
        qml.from_qiskit(transpiled_ansatz)(weights)
        # qml.from_qiskit(transpiled_ansatz)(w)
        return qml.expval(qml.Z(qubit_map[1]))

    # data = np.array([i/2 for i in range(n_wires)], requires_grad=False)
    data = np.array([i / 2 for i in range(n_wires)], requires_grad=False)

    weights = np.array(
        [i / 10 for i in range(ansatz.num_parameters)], requires_grad=True
    )

    print("\tPrepare ideal circuit!")
    ideal_circuit = qml.QNode(pennylane_circuit, dev)

    print("\tPrepare quantum torch layer.")
    n_trainable = len(ansatz.parameters)

    weight_shapes = {"weights": [n_trainable]}

    quantum_layer = qml.qnn.TorchLayer(ideal_circuit, weight_shapes)

    # drawer = qml.draw(ideal_circuit)
    # print(drawer(data, weights))
    print("\tCheck if the quatum torch layer works!")
    try:
        data = Tensor(data)
        data = data.to("cpu")
        print(f"\t\t{quantum_layer(data)}")
    except TypeError as e:
        print(
            "\tWe expect a TypeError to occur, because IQM-native R gate is not native for Pennylane. See:"
        )
        print(
            "\t\thttps://discuss.pennylane.ai/t/how-to-train-a-circuit-imported-from-qiskit/1832"
        )
        print(
            "\t\thttps://discuss.pennylane.ai/t/load-parametrized-qiskit-instructions-that-act-on-multiple-qubits/1801/5"
        )
        print(
            "\t\thttps://docs.quantum.ibm.com/api/qiskit/qiskit.circuit.library.RGate"
        )
        print(
            "\t\thttps://pennylane.ai/blog/2021/05/how-to-add-custom-gates-and-templates-to-pennylane"
        )
        print(f"\tThe error is:\n\t{e}")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
