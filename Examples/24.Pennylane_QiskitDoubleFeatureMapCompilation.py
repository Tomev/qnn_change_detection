# Let's see if I can create Pennylane circuit from transpiled qiskit circuit.

from typing import Dict

import pennylane as qml
import torch.nn as nn
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


def main() -> None:

    # https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qiskit.html
    n_wires = 8
    dev = qml.device("default.qubit")

    data = ParameterVector("inputs", n_wires)
    data2 = ParameterVector("inputs", n_wires)

    feature_map = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map.ry(data[i], i)

    feature_map2 = QuantumCircuit(n_wires)

    for i in range(n_wires):
        feature_map2.ry(data2[i], i)

    # backend = provider.get_backend('ibm_brisbane')
    simulator = FakeBrisbane()

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    tqc = pm.run(feature_map)
    tqc2 = pm.run(feature_map2)

    # print(tqc)

    @qml.qnode(dev)
    def circuit(angles, angles2):
        qml.from_qiskit(tqc)(angles)
        qml.from_qiskit(tqc2)(angles2)
        return qml.expval(qml.Z(0))

    angles = [i for i in range(8)]
    angles2 = [i for i in range(8, 0, -1)]
    print(circuit(angles, angles2))
    print("\n\n")

    # Check if there's the qiskit->qasm->qiskit compilation needed to remove unused qubits.
    drawer = qml.draw(circuit)
    print(drawer(angles, angles2))
    # It's not!


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
