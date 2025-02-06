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

    angles = ParameterVector("angles", 2)

    dev = qml.device("default.qubit")

    qc = QuantumCircuit(2, 2)
    qc.rx(angles[0], 0)
    qc.ry(angles[1], 1)
    qc.cx(1, 0)

    # backend = provider.get_backend('ibm_brisbane')
    simulator = FakeBrisbane()

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    tqc = pm.run(qc)

    print(tqc)

    @qml.qnode(dev)
    def circuit(angles):
        qml.from_qiskit(tqc)(angles)
        return qml.expval(qml.Z(0))

    angles = [3.1, 0.45]
    print(circuit(angles))


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
