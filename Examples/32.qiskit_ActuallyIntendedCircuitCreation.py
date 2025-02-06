from typing import Dict

import pennylane as qml
from pennylane import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import PauliTwoDesign
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import (FakeBrisbane, FakeMelbourne,
                                              FakeMelbourneV2)
from qiskit_machine_learning.utils import algorithm_globals


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


def create_pennylane_circuit(fm_tqc, a_tqc):

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
    algorithm_globals.random_seed = 42
    n_qubits = n_wires = 8

    ansatz = SimplifiedTwoDesignQiskit(n_qubits=n_qubits, n_layers=3)

    # print(ansatz)

    # Ansatz transpilation

    # backend = provider.get_backend('ibm_brisbane')
    simulator = FakeBrisbane()

    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    transpiled_ansatz = pm.run(ansatz)

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(f"Permutation map: {qubit_map}")

    # transpiled_ansatz_qasm = dumps(transpiled_ansatz)
    # transpiled_ansatz = parse(transpiled_ansatz_qasm)

    print(transpiled_ansatz)

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

    p_qc = create_pennylane_circuit(transpiled_feature_map, transpiled_ansatz)

    data = np.array([i for i in range(n_wires)])
    weights = np.array(
        [i / 10 for i in range(ansatz.num_parameters)], requires_grad=True
    )

    # print("\tPrepare ideal circuit!")
    # ideal_circuit = qml.QNode(p_qc, dev)

    drawer = qml.draw(p_qc)
    print(drawer(data, weights))


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
