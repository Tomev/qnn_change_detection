from typing import Dict

import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliTwoDesign
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2
from qiskit_ibm_runtime.fake_provider import FakeMelbourneV2
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from torch import Tensor


def layout_to_map(layout, quantum_register) -> Dict[int, int]:
    # Get simple qubit map from circuit layout.
    # The map is a dict in form {virtual -> physical}.
    # We want to act on the physical qubits.
    initial_layout = layout.initial_layout.get_virtual_bits()
    # print(test)

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

    ansatz = PauliTwoDesign(n_wires, reps=1)

    # Ansatz transpilation

    melbournev2_backend = FakeMelbourneV2()
    pm = generate_preset_pass_manager(optimization_level=3, backend=melbournev2_backend)
    transpiled_ansatz = pm.run(ansatz)

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(f"Permutation map: {qubit_map}")

    # Feature map creation
    print(f"Number of qubit in transpiled circuits: {transpiled_ansatz.num_qubits}")
    feature_map = QuantumCircuit(transpiled_ansatz.num_qubits)
    input_params = []

    for k in qubit_map:
        input_params.append(Parameter(f"input{k}"))
        feature_map.ry(input_params[-1], qubit_map[k])

    # Intended observables:
    # In the initial experiment (pennylane) we use Z on first qubit.
    observable = SparsePauliOp.from_sparse_list(
        [("Z", [qubit_map[1]], 1)], num_qubits=transpiled_ansatz.num_qubits
    )

    # Network creation
    qnn_circuit = QNNCircuit(feature_map=feature_map, ansatz=transpiled_ansatz)
    # There's an DeprecationWarning concerning Estimator (default is V1). We will
    # deal with this in another example.
    # estimator = estimator=EstimatorV2()

    estimator_qnn = EstimatorQNN(
        circuit=qnn_circuit,
        observables=observable,
    )

    # Sample weights and inputs
    estimator_qnn_input = algorithm_globals.random.random(estimator_qnn.num_inputs)
    estimator_qnn_weights = algorithm_globals.random.random(estimator_qnn.num_weights)

    print(f"\n\nInputs:\n\tLen: {len(estimator_qnn_input)}\n\t{estimator_qnn_input}")

    print(
        f"\n\nWeights:\n\tLen: {len(estimator_qnn_weights)}\n\t{estimator_qnn_weights}"
    )

    # Forward pass example
    estimator_qnn_forward = estimator_qnn.forward(
        estimator_qnn_input, estimator_qnn_weights
    )

    print(
        f"Forward pass result for EstimatorQNN: {estimator_qnn_forward}. \nShape: {estimator_qnn_forward.shape}"
    )

    # Batch forward pass
    # For the EstimatorQNN, the expected output shape for the forward pass is (batch_size, num_qubits * num_observables)
    estimator_qnn_forward_batched = estimator_qnn.forward(
        [estimator_qnn_input, estimator_qnn_input], estimator_qnn_weights
    )

    print(
        f"Forward pass result for EstimatorQNN: {estimator_qnn_forward_batched}.  \nShape: {estimator_qnn_forward_batched.shape}"
    )

    # Backward pass
    estimator_qnn_input_grad, estimator_qnn_weight_grad = estimator_qnn.backward(
        estimator_qnn_input, estimator_qnn_weights
    )

    print(
        f"Input gradients for EstimatorQNN: {estimator_qnn_input_grad}.  \nShape: {estimator_qnn_input_grad}"
    )
    print(
        f"Weight gradients for EstimatorQNN: {estimator_qnn_weight_grad}.  \nShape: {estimator_qnn_weight_grad.shape}"
    )


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
