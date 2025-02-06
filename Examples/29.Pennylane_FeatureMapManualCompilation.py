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


def create_pennylane_circuit(fm_tqc, a_tqc):

    gate_map = {"sx": qml.SX, "rz": qml.RZ}

    def circuit(inputs, weights):

        def get_param(p):

            # Normal parameter
            if p is None or isinstance(p, float):
                return p

            # Parameter expression
            param = p["addend"]

            if p["name"] == "inputs":
                param += inputs[p["index"]]
            else:
                param += weights[p["index"]]

            return param

        for gate in fm_tqc:
            g_info = gate_info(gate)

            p = get_param(g_info["param"])

            if p:
                gate_map[g_info["name"]](p, g_info["qubits"])
            else:
                gate_map[g_info["name"]](g_info["qubits"])

    return circuit


def dissasemble_parameter_expression(exp):
    # For simplicity I'll assume:
    # - The only sympy operation is addition
    # - The order is NUMBER + PARAMETER
    addends = str(exp._symbol_expr).split(" + ")

    return {
        "name": addends[1].split("[")[0],
        "index": int(addends[1].split("[")[1].replace("]", "")),
        "addend": float(addends[0]),
    }


def gate_info(gate):

    qubits = []

    for q in gate[1]:
        qubits.append(q._index)

    param = None

    from qiskit.circuit import ParameterExpression

    for p in gate[0].params:
        param = p
        if isinstance(p, ParameterExpression):
            param = dissasemble_parameter_expression(p)

    return {"name": gate[0].name, "qubits": qubits, "param": param}


def main() -> None:

    # https://docs.pennylane.ai/en/stable/code/api/pennylane.from_qiskit.html

    dev = qml.device("default.qubit")

    feature_map = QuantumCircuit(8)

    data = ParameterVector("inputs", 8)

    for i in range(8):
        feature_map.ry(data[i], i)

    # backend = provider.get_backend('ibm_brisbane')
    simulator = FakeBrisbane()

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    tqc = pm.run(feature_map)

    # print(tqc)

    p_qc = create_pennylane_circuit(tqc, None)

    drawer = qml.draw(p_qc)
    print(drawer([i for i in range(8)], []))


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
