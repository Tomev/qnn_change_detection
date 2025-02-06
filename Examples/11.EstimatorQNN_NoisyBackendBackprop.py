from typing import Dict

import torch.nn as nn
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliTwoDesign
from qiskit.qasm3 import dumps
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import Estimator
from qiskit_ibm_runtime.fake_provider import (FakeBrisbane, FakeMelbourne,
                                              FakeMelbourneV2)
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
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

    ansatz = PauliTwoDesign(n_wires, reps=1)

    # Ansatz transpilation

    # backend = provider.get_backend('ibm_brisbane')
    # simulator = AerSimulator.from_backend(backend)
    simulator = FakeBrisbane()
    # simulator = FakeMelbourne() # Works
    simulator = FakeMelbourneV2()  # Works
    pm = generate_preset_pass_manager(optimization_level=2, backend=simulator)
    transpiled_ansatz = pm.run(ansatz)

    # print(transpiled_ansatz)

    # return

    qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)
    print(f"Permutation map: {qubit_map}")

    transpiled_ansatz_qasm = dumps(transpiled_ansatz)
    transpiled_ansatz = parse(transpiled_ansatz_qasm)

    # print(transpiled_ansatz)

    # Feature map creation
    print(f"Number of qubit in transpiled circuits: {transpiled_ansatz.num_qubits}")
    feature_map = QuantumCircuit(transpiled_ansatz.num_qubits)
    input_params = []

    for k in qubit_map:
        input_params.append(Parameter(f"input{k}"))
        feature_map.ry(input_params[-1], qubit_map[k])

    # Transpilation is needed, to ensure proper gates are used.
    # Ensure optimization level is low, so that the qubits are not permuted.
    pm = generate_preset_pass_manager(optimization_level=0, backend=simulator)
    transpiled_feature_map = pm.run(feature_map)
    transpiled_feature_map_qasm = dumps(transpiled_feature_map)
    transpiled_feature_map = parse(transpiled_feature_map_qasm)

    # Intended observables:
    # In the initial experiment (pennylane) we use Z on first qubit.
    observable = SparsePauliOp.from_sparse_list(
        [("Z", [qubit_map[1]], 1)], num_qubits=transpiled_ansatz.num_qubits
    )

    # Network creation
    qnn_circuit = QNNCircuit(
        feature_map=transpiled_feature_map, ansatz=transpiled_ansatz
    )
    # There's an DeprecationWarning concerning Estimator (default is V1). We will
    # deal with this in another example.

    estimator = Estimator(simulator)
    gradient = ParamShiftEstimatorGradient(estimator)

    # gradient = estimator = None

    estimator_qnn = EstimatorQNN(
        circuit=qnn_circuit,
        observables=observable,
        estimator=estimator,
        input_params=transpiled_feature_map.parameters,
        weight_params=transpiled_ansatz.parameters,
        input_gradients=True,
        gradient=gradient,
    )

    # Sample weights and inputs
    estimator_qnn_input = algorithm_globals.random.random(estimator_qnn.num_inputs)
    estimator_qnn_weights = algorithm_globals.random.random(estimator_qnn.num_weights)

    print(f"\n\nInputs:\n\tLen: {len(estimator_qnn_input)}\n\t{estimator_qnn_input}")

    print(
        f"\n\nWeights:\n\tLen: {len(estimator_qnn_weights)}\n\t{estimator_qnn_weights}"
    )

    # Backward pass
    try:
        estimator_qnn_input_grad, estimator_qnn_weight_grad = estimator_qnn.backward(
            estimator_qnn_input, estimator_qnn_weights
        )

        print(
            f"Input gradients for EstimatorQNN: {estimator_qnn_input_grad}.  \nShape: {estimator_qnn_input_grad}"
        )
        print(
            f"Weight gradients for EstimatorQNN: {estimator_qnn_weight_grad}.  \nShape: {estimator_qnn_weight_grad.shape}"
        )
    except QiskitMachineLearningError as e:
        print("As far as I know we cannot compute gradients on hardware simulators.")
        print("That also means, we cannot do noisy training with qiskit.")
        print(f"Indeed, we found {e}.")
        print("This has to be a problem with the gradient computation for the device.")
        print("Assuming, that is, on the behavior of the constructor.")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
