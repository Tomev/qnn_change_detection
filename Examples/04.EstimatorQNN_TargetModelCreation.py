import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliTwoDesign
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import EstimatorV2
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from torch import Tensor


def main() -> None:

    # https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
    # Example implementation.

    # Config
    algorithm_globals.random_seed = 42
    n_qubits = 4
    n_wires = 2 * n_qubits

    # Network creation
    feature_map = QuantumCircuit(n_wires)
    input_params = []

    for i in range(n_wires):
        input_params.append(Parameter(f"input{i}"))
        feature_map.ry(input_params[-1], i)

    ansatz = PauliTwoDesign(n_wires, reps=1)

    observable = SparsePauliOp.from_sparse_list([("Z", [0], 1)], num_qubits=n_wires)

    qnn_circuit = QNNCircuit(feature_map=feature_map, ansatz=ansatz)

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
