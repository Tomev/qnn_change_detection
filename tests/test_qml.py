import unittest
from abc import ABC

import pennylane as qml
import torch
from torch import Tensor, rand

from src.quantum_classifier import MCQuantumNeuralNetwork, QuantumNeuralNetwork


class TestQNNBase(unittest.TestCase, ABC):
    """
    QNNBase models have the same functionalities to chech, so it's smarter to
    prapare the base class also for the tests.
    """

    def setUp(self) -> None:
        self.n_wires: int = 8  # Hard coded for our current models.
        self.torch_dev = "cpu"

        self.I1: Tensor = rand(self.n_wires // 2)
        self.I2: Tensor = rand(self.n_wires // 2)

        self.X: Tensor = torch.cat((self.I1, self.I2), 0)
        self.X = torch.flatten(self.X).to(self.torch_dev)

        super().setUp()

    def tearDown(self) -> None:
        super().tearDown()


class TestQuantumNeuralNetwork(TestQNNBase):

    def setUp(self) -> None:
        super().setUp()

        self.dev = qml.device("default.qubit")
        self.model = QuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

    def test_model_forward(self):
        self.model(self.I1, self.I2)
        assert True


class TestMCQuantumNeuralNetwork(TestQNNBase):

    def setUp(self) -> None:
        super().setUp()

        self.dev = qml.device("default.qubit")
        self.model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

    def test_model_forward(self):
        self.model.forward_single(self.X)
        assert True

    def test_different_models_outputs(self):
        other_model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

        model_out = self.model.forward_single(self.X)
        other_model_out = other_model.forward_single(self.X)

        assert torch.not_equal(model_out, other_model_out)

    def test_different_weights_on_model_init(self):
        other_model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

        model_weights = self.model.get_weights()
        other_model_weights = other_model.get_weights()

        for i in range(len(model_weights)):
            assert not torch.allclose(model_weights[i], other_model_weights[i])

    def test_same_sample_output(self):
        o1 = self.model.forward_single(self.X)
        o2 = self.model.forward_single(self.X)

        assert o1 == o2

    def test_weights_list_application(self):

        init_params = self.model.parameters_list()

        other_model = other_model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

        self.model.apply_weights_list(other_model.parameters_list())

        o = self.model.forward_single(self.X)
        o_other = other_model.forward_single(self.X)

        # Check that they did change
        assert init_params != self.model.parameters_list()
        # Check if they changed to the right value
        assert self.model.parameters_list() == other_model.parameters_list()
        # We have to also check outputs, because we had the issue when the
        # weights values changed, but weren't used by the model.
        assert o == o_other

    def test_weights_application(self):
        init_params = self.model.get_weights()

        other_model = other_model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

        self.model.apply_weights(other_model.get_weights())

        o = self.model.forward_single(self.X)
        o_other = other_model.forward_single(self.X)

        # Check that they did change
        assert init_params != self.model.parameters_list()
        # Check if they changed to the right value
        assert self.model.parameters_list() == other_model.parameters_list()
        # We have to also check outputs, because we had the issue when the
        # weights values changed, but weren't used by the model.
        assert o == o_other

    def test_state_dict_load(self):

        other_model = other_model = MCQuantumNeuralNetwork(
            self.n_wires // 2, quantum_dev=self.dev, torch_device=self.torch_dev
        )

        other_model.load_state_dict(self.model.state_dict())

        o1 = self.model.forward_single(self.X)
        o2 = other_model.forward_single(self.X)

        assert o1 == o2


if __name__ == "__main__":
    unittest.main()
