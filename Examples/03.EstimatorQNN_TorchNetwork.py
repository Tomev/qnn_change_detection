import torch.nn as nn
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.utils import algorithm_globals
from torch import Tensor


def main() -> None:

    # https://qiskit-community.github.io/qiskit-machine-learning/tutorials/01_neural_networks.html
    # Example implementation.

    # Network creation
    algorithm_globals.random_seed = 42

    params1 = [Parameter("input1"), Parameter("weight1")]
    qc1 = QuantumCircuit(1)
    qc1.h(0)
    qc1.ry(params1[0], 0)
    qc1.rx(params1[1], 0)
    # Skip the drawing, for now.
    # qc1.draw("mpl", style="clifford")

    observable1 = SparsePauliOp.from_list([("Y" * qc1.num_qubits, 1)])

    estimator_qnn = EstimatorQNN(
        circuit=qc1,
        observables=observable1,
        input_params=[params1[0]],
        weight_params=[params1[1]],
    )

    # Sample weights and inputs
    estimator_qnn_input = algorithm_globals.random.random(estimator_qnn.num_inputs)
    estimator_qnn_weights = algorithm_globals.random.random(estimator_qnn.num_weights)

    # Batch forward pass
    # For the EstimatorQNN, the expected output shape for the forward pass is (batch_size, num_qubits * num_observables)
    estimator_qnn_batch_input = [estimator_qnn_input, estimator_qnn_input]

    # It's also possible to set gradients for input --- I skipped that.
    torch_input = Tensor(estimator_qnn_input)
    torch_batch_input = Tensor(estimator_qnn_batch_input)

    quantum_layer = TorchConnector(
        neural_network=estimator_qnn, initial_weights=estimator_qnn_weights
    )

    classical_layer = nn.Linear(1, 1)
    sigmoid_layer = nn.Sigmoid()

    model = nn.Sequential(classical_layer, quantum_layer, sigmoid_layer)

    print(f"\n\nWhole model forward results {model.forward(torch_input)}")

    print(f"\n\nWhole model batch forward results {model.forward(torch_batch_input)}")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
