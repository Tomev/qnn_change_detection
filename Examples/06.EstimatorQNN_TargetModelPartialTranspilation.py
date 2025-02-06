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

    # Transpilation
    melbournev2_backend = FakeMelbourneV2()
    pm = generate_preset_pass_manager(optimization_level=3, backend=melbournev2_backend)
    transpiled_ansatz = pm.run(ansatz)

    print("\n\n\n\n")

    print(ansatz)

    print("\n\n\n\n")

    print(transpiled_ansatz)

    print("\n\n\n\n")

    print(f"Layout: {transpiled_ansatz.layout}")

    print("Ansatz transpiled.\n")


if __name__ == "__main__":
    print("Experiment start.")
    main()
    print("Experiment finish.")
