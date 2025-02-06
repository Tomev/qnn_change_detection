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


def create_pennylane_circuit(fm_tqc, a_tqc, qubit_map):

    gate_map = {"sx": qml.SX, "rz": qml.RZ, "ecr": qml.ECR, "x": qml.X}

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

        for tqc in [fm_tqc, a_tqc]:
            # for tqc in [fm_tqc]:
            # for tqc in [a_tqc]:
            for gate in tqc:

                # print(tqc)

                g_info = gate_info(gate)

                # print(g_info)

                # For some reason there are "NONE" gates.
                if None in g_info["qubits"]:
                    continue

                p = get_param(g_info["param"])

                if p is None:
                    gate_map[g_info["name"]](g_info["qubits"])
                else:
                    gate_map[g_info["name"]](p, g_info["qubits"])

        return qml.expval(qml.PauliZ(qubit_map[1]))

    return circuit


def dissasemble_parameter_expression(exp):
    # For simplicity I'll assume:
    # - The only sympy operation is addition
    # - The order is NUMBER + PARAMETER
    addends = str(exp._symbol_expr).split(" + ")

    if len(addends) == 1:
        addends = [0.0, addends[0]]

    # print(addends)
    name = addends[1].split("[")[0]

    index = int(addends[1].split("[")[1].replace("]", ""))

    return {"name": name, "index": index, "addend": float(addends[0])}


def gate_info(gate):

    # print(gate)

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
    simulator = FakeBrisbane()

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    transpiled_ansatz = pm.run(ansatz)

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(f"Permutation map: {qubit_map}")

    # transpiled_ansatz_qasm = dumps(transpiled_ansatz)
    # transpiled_ansatz = parse(transpiled_ansatz_qasm)

    # print(transpiled_ansatz)

    # Feature map creation
    print(f"Number of qubit in transpiled circuits: {transpiled_ansatz.num_qubits}")
    feature_map = QuantumCircuit(simulator.num_qubits)

    data = ParameterVector("inputs", n_wires)

    for i, k in enumerate(qubit_map.keys()):
        feature_map.ry(data[i], qubit_map[k])

    # Transpilation is needed, to ensure proper gates are used.
    # Ensure optimization level is low, so that the qubits are not permuted.
    pm = generate_preset_pass_manager(optimization_level=0, backend=simulator)
    transpiled_feature_map = pm.run(feature_map)

    # Pennylane transpilaiton
    # There's bug when compiling ECR gates via pennylane_qiskit.converter.
    # TODO: Analyze and submit it later. It can be easily fixed.
    # dev = qml.device("default.mixed", wires=qubit_map.values())  # Notice default.mixed
    dev = qml.device("default.qubit")

    p_qc = create_pennylane_circuit(
        transpiled_feature_map, transpiled_ansatz, qubit_map
    )

    data = np.array([i / 2.0 for i in range(n_wires)], requires_grad=False)
    weights = np.array(
        [i / 10 for i in range(ansatz.num_parameters)], requires_grad=True
    )

    # print("\tPrepare ideal circuit!")
    # ideal_circuit = qml.QNode(p_qc, dev)

    # drawer = qml.draw(p_qc)
    # print(drawer(data, weights))
    ideal_circuit = qml.QNode(p_qc, dev)

    print("\n\n\tCircuit execution:")
    print(f"\t\tIdeal: {ideal_circuit(data, weights)}")

    print("\n\nPrepare training-related stuff.")
    print(f"\tInitial weights:\n\t\t{data}\n\t\t{weights}")
    opt = qml.GradientDescentOptimizer()

    print("\n\n\tGradient computation:")
    weights, _ = opt.step_and_cost(ideal_circuit, data, weights)
    print(f"\tFinal weights:\n\t\t{weights}")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
