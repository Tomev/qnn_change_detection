# Since the wires numbering in the Pennylane circuit compiled from the qiskit circuit
# is the same as on the device, it might be possible for us to use the noise model
# that is also automatically obtained.

from typing import Dict

import pennylane as qml
import torch.nn as nn
from pennylane_qiskit.converter import load_noise_model
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
    print("\tGet backend!")
    service = QiskitRuntimeService()
    backend = service.backend("ibm_brisbane")

    print("\tExtract qiskit noise model!")
    model = NoiseModel.from_backend(backend)

    print("\tPrepare pennylane model!")
    pl_model = load_noise_model(model)

    print(pl_model)


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
