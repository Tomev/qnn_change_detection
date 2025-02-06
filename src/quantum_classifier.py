from abc import abstractmethod
from math import prod
from typing import Any, Callable, Dict, List, Union

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from iqm.qiskit_iqm.iqm_provider import IQMProvider
from pennylane.devices import Device
from pennylane.measurements import ExpectationMP
from qiskit import QuantumCircuit
from qiskit.circuit import (Instruction, Parameter, ParameterExpression,
                            ParameterVector, Qubit)
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import TranspileLayout
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.circuit.library import QNNCircuit
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.gradients import ParamShiftEstimatorGradient
from qiskit_machine_learning.neural_networks import EstimatorQNN
from torch import Tensor, tensor


def layout_to_map(
    layout: TranspileLayout, quantum_register: List[Qubit]
) -> Dict[int, int]:
    # Get simple qubit map from circuit layout.
    # The map is a dict in form {virtual -> physical}.
    # We want to act on the physical qubits.
    initial_layout = layout.initial_layout.get_virtual_bits()

    qubit_map = {}

    for k in quantum_register:
        qubit_map[k._index] = initial_layout[k]

    return qubit_map


class BellmanLayer(qml.operation.Operation):

    num_wires = qml.operation.AnyWires
    grad_method = None

    def __init__(self, weights, wires, id=None):
        # TR: I can add some additional checks here.
        super().__init__(weights, wires=wires, id=id)

    @staticmethod
    def compute_decomposition(weights, wires):
        n_layers = qml.math.shape(weights)[0]
        op_list = []

        for layer in range(n_layers):

            op_list.append(qml.H(wires=wires[0]))

            for i in range(len(wires) - 1):
                op_list.append([qml.CNOT(wires=[wires[i], wires[i + 1]])])

            for i in range(len(wires)):
                op_list.append(qml.RY(phi=weights[layer][i], wires=wires[i]))

            for i in range(len(wires) - 1, 0, -1):
                op_list.append([qml.CNOT(wires=[wires[i - 1], wires[i]])])

        return op_list

    @staticmethod
    def shape(n_layers, n_wires):
        r"""Returns a list of shapes for the 2 parameter tensors.

        Args:
            n_layers (int): number of layers
            n_wires (int): number of wires

        Returns:
            list[tuple[int]]: list of shapes
        """
        return [(n_wires,), (n_layers, n_wires)]


def SimplifiedTwoDesignQiskit(n_qubits: int, n_layers: int) -> QuantumCircuit:
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

    c: QuantumCircuit = QuantumCircuit(n_qubits)

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


# TODO TR: This is more of a hybrid NN, not a Quantum one, as there are
#           classical layers in it. Ensure that it's the case and correct
#           the name of the class if necessary.


class QNNBase(nn.Module):
    """
    Base class for our hybrid models.
    """

    def __init__(
        self, n_qubits: int, quantum_dev: Device, torch_device: str = "cpu"
    ) -> None:
        super().__init__()
        self.quantum_dev: Device = quantum_dev
        self.torch_device = torch_device
        self.n_qubits: int = n_qubits

        self.classical_linear_layer: nn.Module = torch.nn.Linear(
            2 * n_qubits, 2 * n_qubits
        )
        self.quantum_layer: nn.Module
        self.prepare_quantum_layer()
        self.sm: nn.Module = nn.Sigmoid()

    @abstractmethod
    def prepare_quantum_layer(self) -> None:
        raise NotImplementedError("Quantum layer not implemented.")

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x: Tensor = torch.cat((x1, x2), 1)
        # TODO TR: Why is there this inter-device conversion? To cuda, to cpu, to cuda.
        x = torch.flatten(x, 1).to(self.torch_device)
        x = self.classical_linear_layer(x).to("cpu")
        x = self.quantum_layer(x).to(self.torch_device)

        output: Tensor = self.sm(x)
        output = torch.flatten(output)
        return output

    def forward_single(self, x: Tensor) -> Tensor:

        x = self.classical_linear_layer(x).to("cpu")
        x = self.quantum_layer(x)
        x = x.to(self.torch_device)
        output: Tensor = self.sm(x)

        return torch.flatten(output)

    def get_weights_sizes(self) -> List[torch.Size]:
        """
        Returns a list of sizes of the weights of each parametrized layer in the
        model. The order of the sizes is the same as of the layers in the model.
        This is important for the sake of apply_weights method.

        :note:
            Biases are also considered as weights in this context.
        """
        sizes: List[torch.Size] = []

        for layer in [self.classical_linear_layer, self.quantum_layer]:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    sizes.append(p.size())

        return sizes

    def get_weights(self):
        weights: List[torch.nn.Parameter] = []

        for layer in [self.classical_linear_layer, self.quantum_layer]:
            if hasattr(layer, "parameters"):
                for p in layer.parameters():
                    weights.append(p.data)

        return weights

    def apply_weights(self, weights):
        for layer in [self.classical_linear_layer, self.quantum_layer]:
            if not hasattr(layer, "parameters"):
                continue
            for p in layer.parameters():
                p.data = nn.parameter.Parameter(weights.pop(0))

    def apply_weights_list(self, weights: List[float]) -> None:
        """
        Applies given weights to the model. This is just a simple substitution
        of the layers parameters thanks to the initial weights preprocessing. See
        _prepare_weights method for more details.
        """
        n_weights: int = sum([prod(s) for s in self.get_weights_sizes()])
        assert (
            len(weights) == n_weights
        ), f"Expected {n_weights} weights, got {len(weights)}."

        # Transform weights to the
        processed_weights: List[torch.Tensor] = self._prepare_weights(weights)
        self.apply_weights(processed_weights)

    def _prepare_weights(self, weights: List[float]) -> List[torch.Tensor]:
        """
        Dissects the weights list into the list of torch tensors of the same sizes
        as the model weights. This is necessary for easy application of the weights.

        :returns:
            List of weights as torch tensors of the same sizes as the model weights.
        """
        weights_sizes: List[torch.Size] = self.get_weights_sizes()
        chunks: List[int] = [prod(s) for s in weights_sizes]

        processed_weights: List[torch.Tensor] = []

        for i in range(len(chunks)):
            processed_weights.append(
                tensor(
                    weights[: chunks[i]], dtype=torch.float32, requires_grad=False
                ).reshape(weights_sizes[i])
            )
            weights = weights[chunks[i] :]

        return processed_weights

    def parameters_list(self) -> List[float]:
        """
        Returns a list of all the parameters of the model. This is useful for
        the sake of optimization algorithms, which require a flat list of parameters
        to optimize.
        """
        return [
            v for w in self.parameters() for v in w.flatten().detach().cpu().numpy()
        ]


class QuantumNeuralNetwork(QNNBase):

    def __init__(
        self, n_qubits: int, quantum_dev: Device, torch_device: str = "cpu"
    ) -> None:
        super(QuantumNeuralNetwork, self).__init__(n_qubits, quantum_dev, torch_device)

    # TR: Final layer, to be restored by the end of the experiments.
    """
    def prepare_quantum_layer(self) -> None:

        qnode = qml.QNode(self._circuit, self.quantum_dev)

        shape = qml.SimplifiedTwoDesign.shape(
            n_layers=3, n_wires=int(2 * self.n_qubits)
        )[1]
        weight_shapes = {"weights": shape}

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def _circuit(self, inputs: Tensor, weights: Tensor) -> ExpectationMP:

        qml.templates.AngleEmbedding(
            inputs, wires=range(2 * self.n_qubits), rotation="Y"
        )

        qml.SimplifiedTwoDesign(
            initial_layer_weights=[0, 0, 0, 0, 0, 0, 0, 0],
            weights=weights,
            wires=range(int(2 * self.n_qubits)),
        )
        return qml.expval(qml.PauliZ(1))
    """

    def prepare_quantum_layer(self) -> None:

        qnode = qml.QNode(self._circuit, self.quantum_dev)

        shape = qml.SimplifiedTwoDesign.shape(
            n_layers=3, n_wires=int(2 * self.n_qubits)
        )[1]

        weight_shapes = {"weights": shape}

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def _circuit(self, inputs: Tensor, weights: Tensor) -> ExpectationMP:

        qml.templates.AngleEmbedding(
            inputs, wires=range(2 * self.n_qubits), rotation="Y"
        )

        qml.SimplifiedTwoDesign(
            initial_layer_weights=[0, 0, 0, 0, 0, 0, 0, 0],
            weights=weights,
            wires=range(int(2 * self.n_qubits)),
        )

        return qml.expval(qml.PauliZ(1))


class MCQuantumNeuralNetwork(QNNBase):
    """
    This is Manually Compiled Quantum Neural Network classifier.
    """

    def __init__(
        self,
        n_qubits: int,
        quantum_dev: Device,
        torch_device: str = "cpu",
        target_backend: str = "ibm_brisbane",
    ) -> None:
        self.target_backend: str = target_backend
        super(MCQuantumNeuralNetwork, self).__init__(
            n_qubits, quantum_dev, torch_device
        )

    def prepare_quantum_layer(self) -> None:

        print("\tGet backend!")
        service = QiskitRuntimeService()
        backend = service.backend(self.target_backend)

        n_wires: int = 2 * self.n_qubits

        print("\tTranspile anastz for selected backend.")
        pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
        ansatz = SimplifiedTwoDesignQiskit(n_qubits=n_wires, n_layers=3)
        transpiled_ansatz = pm.run(ansatz)

        print("\tTranspile feature map for selected backend.")
        qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)

        feature_map = QuantumCircuit(backend.num_qubits)

        data = ParameterVector("inputs", n_wires)  # Name "inputs" is important!

        for i, k in enumerate(qubit_map.keys()):
            feature_map.ry(data[i], qubit_map[k])

        pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
        transpiled_feature_map = pm.run(feature_map)

        print("\tCreate QNode.")

        # self.dev = qml.device("default.mixed", wires=qubit_map.values())

        circuit = self.create_pennylane_circuit(
            transpiled_feature_map, transpiled_ansatz, qubit_map
        )

        qnode = qml.QNode(circuit, self.quantum_dev)
        n_trainable = len(ansatz.parameters)

        weight_shapes = {"weights": [n_trainable]}

        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def create_pennylane_circuit(
        self, fm_tqc: QuantumCircuit, a_tqc: QuantumCircuit, qubit_map: Dict[int, int]
    ) -> Callable[[Tensor, Tensor], ExpectationMP]:

        # Gate map is device dependent! This works for IBM_BRISBANE only!
        gate_map = {"sx": qml.SX, "rz": qml.RZ, "ecr": qml.ECR, "x": qml.X}

        # @partial(qml.batch_input, argnum=1)
        def circuit(inputs: Tensor, weights: Tensor) -> ExpectationMP:

            def get_param(p: Union[float, Dict[str, Any]]) -> float:

                # Normal parameter
                if p is None or isinstance(p, float):
                    return p

                # Parameter expression
                param: float = p["addend"]

                if p["name"] == "inputs":
                    param += float(inputs[p["index"]])
                else:
                    param += float(weights[p["index"]])

                return param

            for tqc in [fm_tqc, a_tqc]:
                for gate in tqc:

                    g_info = self.gate_info(gate)

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

    def dissasemble_parameter_expression(
        self, expression: ParameterExpression
    ) -> Dict[str, Any]:
        # For simplicity I'll assume:
        # - The only sympy operation is addition
        # - The order is NUMBER + PARAMETER
        addends: List[str] = str(expression._symbol_expr).split(" + ")

        if len(addends) == 1:
            addends = ["0.0", addends[0]]

        # print(addends)
        name = addends[1].split("[")[0]

        index = int(addends[1].split("[")[1].replace("]", ""))

        return {"name": name, "index": index, "addend": float(addends[0])}

    def gate_info(self, gate: Instruction) -> Dict[str, Any]:

        qubits = []

        for q in gate[1]:
            qubits.append(q._index)

        param = None

        for p in gate[0].params:
            param = p
            if isinstance(p, ParameterExpression):
                param = self.dissasemble_parameter_expression(p)

        return {"name": gate[0].name, "qubits": qubits, "param": param}


class QiskitQuantumNeuralNetwork(QNNBase):
    """
    An implementation of our quantum classifier using QiskitMachineLearning.
    """

    def __init__(
        self, n_qubits: int, quantum_dev: Device, torch_device: str = "cpu"
    ) -> None:
        super(QiskitQuantumNeuralNetwork, self).__init__(
            n_qubits, quantum_dev, torch_device
        )

    def prepare_quantum_layer(self) -> None:

        n_wires = 2 * self.n_qubits

        def create_qnn() -> EstimatorQNN:
            # Ansatz preparation

            # We want to start with ansatz, because it's more complicated than our selected
            # feature map (we use simple AngleEmbedding, which is a layer of single qubit
            # rotations). This way, we can transpile it, get the final layout of our circuit
            # and prepare the feature map and observable according to the layout obtained by
            # ansatz transpilation.
            iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet"
            provider = IQMProvider(iqm_server_url)  # , token=token)
            backend = provider.get_backend()

            ansatz = SimplifiedTwoDesignQiskit(n_qubits=n_wires, n_layers=3)

            # Ansatz transpilation
            pm = generate_preset_pass_manager(optimization_level=2, backend=backend)
            transpiled_ansatz = pm.run(ansatz)

            qubit_map = layout_to_map(transpiled_ansatz.layout, ansatz.qubits)

            # Feature map creation
            print(
                f"Number of qubit in transpiled circuits: {transpiled_ansatz.num_qubits}"
            )
            feature_map = QuantumCircuit(transpiled_ansatz.num_qubits)
            input_params = []

            for k in qubit_map:
                input_params.append(Parameter(f"input{k}"))
                feature_map.ry(input_params[-1], qubit_map[k])

            # Transpilation is needed, to ensure proper gates are used.
            # Ensure optimization level is low, so that the qubits are not permuted.
            pm = generate_preset_pass_manager(optimization_level=0, backend=backend)
            transpiled_feature_map = pm.run(feature_map)

            # In the initial experiment (pennylane) we use Z on first qubit.
            observable = SparsePauliOp.from_sparse_list(
                [("Z", [qubit_map[1]], 1)], num_qubits=transpiled_ansatz.num_qubits
            )

            print(observable)

            # Network creation
            qnn_circuit = QNNCircuit(
                feature_map=transpiled_feature_map, ansatz=transpiled_ansatz
            )

            # sim = AerSimulator().from_backend(backend)  # Noisy
            sim = AerSimulator()
            estimator = Estimator(sim)

            # Prepare to run on real device
            # iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet:mock"  # Remove mock!
            # provider = IQMProvider(iqm_server_url)#, token=token)
            # backend = provider.get_backend()
            # estimator = Estimator(backend)

            gradient = ParamShiftEstimatorGradient(estimator, pass_manager=pm)

            observable = SparsePauliOp.from_sparse_list(
                [("Z", [qubit_map[1]], 1)], num_qubits=transpiled_ansatz.num_qubits
            )

            estimator_qnn = EstimatorQNN(
                circuit=qnn_circuit,
                observables=observable,
                estimator=estimator,
                gradient=gradient,
                input_gradients=False,
            )

            return estimator_qnn

        qnn = create_qnn()

        self.quantum_layer = TorchConnector(qnn)


class Quantum1DCovolution(nn.Module):
    def __init__(self, n_qubits: int) -> None:
        super(Quantum1DCovolution, self).__init__()
        self.dev = qml.device("default.qubit", wires=int(n_qubits))

        def encoding(inputs: Tensor, wires: range) -> None:
            qml.templates.AngleEmbedding(inputs, wires, rotation="X")

        def quantum_convolution(weights: Tensor, wires: List[int]) -> None:
            qml.CRZ(weights[0], wires)
            qml.CRX(weights[1], wires)

        def quantum_pooling(source: int, sink: int) -> None:
            m = qml.measure(sink)
            qml.cond(m, qml.RY)(np.pi / 4, wires=source)

        def qcnn(inputs: Tensor, weights: Tensor) -> ExpectationMP:
            wires: range = range(n_qubits)
            encoding(inputs, wires)
            quantum_convolution(weights[0], wires=[0, 1])
            quantum_pooling(source=0, sink=1)
            quantum_convolution(weights[1], wires=[2, 3])
            quantum_pooling(source=2, sink=3)
            quantum_convolution(weights[2], wires=[4, 5])
            quantum_pooling(source=4, sink=5)
            quantum_convolution(weights[3], wires=[6, 7])
            quantum_pooling(source=6, sink=7)
            quantum_convolution(weights[4], wires=[0, 2])
            quantum_pooling(source=0, sink=2)
            quantum_convolution(weights[5], wires=[4, 6])
            quantum_pooling(source=4, sink=6)
            quantum_convolution(weights[6], wires=[0, 4])
            quantum_pooling(source=0, sink=4)
            return qml.expval(qml.PauliZ(0))

        weight_shapes = {"weights": (7, 2)}

        self.quantum_layer = qml.qnn.TorchLayer(
            qml.QNode(qcnn, self.dev), weight_shapes
        )

    def forward(self, x: Tensor) -> Any:
        return self.quantum_layer(x)


class ConvolutionalQuantumNeuralNetwork(nn.Module):
    def __init__(self, n_channels: int, n_qubits: int) -> None:
        super(ConvolutionalQuantumNeuralNetwork, self).__init__()

        self.torch_device: str = "cuda"

        self.n_channels = n_channels
        self.quantum_convolutional_layer = nn.ModuleList()
        for _ in range(self.n_channels):
            self.quantum_convolutional_layer.append(Quantum1DCovolution(n_qubits))
        self.classical_linear_layer = torch.nn.Linear(self.n_channels, 2)
        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        batch_size = x1.shape[0]
        num_features = 2 * x1.shape[2] * x1.shape[3]

        X = Tensor()

        for i in range(self.n_channels):
            x = torch.dstack(
                (torch.flatten(x1[:, i, :, :], 1), torch.flatten(x2[:, i, :, :], 1))
            )
            x = torch.reshape(x, (batch_size, num_features)).to("cpu")
            x = self.quantum_convolutional_layer[i](x).to(self.torch_device)
            torch.cat((X, x))

        X.transpose_(0, 1)
        # X = torch.dstack(X)
        # X = torch.squeeze(X)

        X = self.classical_linear_layer(X)

        output: Tensor = self.sm(X)

        return output
