from typing import Optional

import torch
import torch.utils
import torch.utils.data
from niapy.algorithms import Algorithm
from niapy.problems import Problem
from niapy.task import OptimizationType, Task
from numpy import pi
from sklearn.metrics import accuracy_score

from src.quantum_classifier import QuantumNeuralNetwork


class QuantumBinaryClassifierTrainingProblem(Problem):
    def __init__(
        self,
        classifier: QuantumNeuralNetwork,
        X_train=None,
        y_train=None,
        data_loader=None,
        dimension: Optional[int] = None,
        lower=-2 * pi,
        upper=2 * pi,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(classifier.n_weights, lower, upper, *args, **kwargs)
        del dimension  # This is not used.
        self.classifier = classifier
        self.X_train = X_train
        self.y_train = y_train
        self.dataloader = data_loader

    def _evaluate(self, solution):
        self.classifier.apply_weights_list(solution)
        return self.classifier.cost(self.classifier.theta, self.X_train, self.y_train)


# TODO TR: Consider inheriting from the lightning trainer.
class MetaheuristicClassifierTrainer:
    """
    This class is responsible for training the quantum classifier using the
    nature inspired metaheuristics from NiaPy.

    TODO TR: Consider adding verbose option.
    """

    def __init__(
        self, algorithm: Algorithm, n_iterations: int = 1000, cutoff: float = 0
    ) -> None:
        """ """
        self.n_iterations: int = n_iterations
        self.algorithm: Algorithm = algorithm
        self.cutoff: float = cutoff

    def get_task(self, classifier, data_loader):
        problem: QuantumBinaryClassifierTrainingProblem = (
            QuantumBinaryClassifierTrainingProblem(
                classifier=classifier, data_loader=data_loader
            )
        )

        task: Task = Task(
            problem=problem, max_iters=self.n_iterations, cutoff_value=self.cutoff
        )

        return task

    def fit(self, classifier: QuantumNeuralNetwork, data_loader):

        task = self.get_task(classifier, data_loader)

        # TODO TR:  Is there an easy way to end the training when the
        #           classifier reaches a certain accuracy? In worst case
        #           scenario we can enforce it on classifier.
        best_solution, best_fitness = self.algorithm.run(task)

        print(f"Best fitness: {best_fitness}")

        classifier.apply_weights_list(best_solution)


class QuantumClassifierTrainingAccuracyProblem(QuantumBinaryClassifierTrainingProblem):
    def _evaluate(self, solution):

        # print("Calling evaluation!")

        self.classifier.apply_weights_list(solution)

        # predictions = self.classifier.predict(self.X_train)

        # Hax for Manish code. To be revisited.
        # predictions = []

        # for x in self.X_train:
        #    predictions.append(self.classifier(x[0], x[1]).detach().round().cpu().numpy())

        # print("Predictions: ", predictions[-1])
        # print("y_train: ", self.y_train[-1].requires_grad)

        # More like Manish code.

        # print("Trying solution: ", solution)

        """ # Evaluation on separate X and y.
        accuracy: float = 0
        data_len: int = 0

        with torch.no_grad():  # So that there's no memory leak.
            for i in range(len(self.X_train)):
                y = self.y_train[i]
                data_len += len(y)

                pred = self.classifier(self.X_train[i][0], self.X_train[i][1])

                accuracy += accuracy_score(
                    y.detach().to(torch.float32).cpu().numpy(),
                    pred.detach().round().cpu().numpy(),
                    normalize=False,
                )

        return accuracy / data_len
        """
        # Dataloader approach
        size = len(self.dataloader.dataset)
        self.classifier.eval()

        correct = 0

        with torch.no_grad():
            for sample in self.dataloader:
                I1, I2, y = sample["I1"], sample["I2"], sample["label"]
                I1 = I1.to("cpu")
                I2 = I2.to("cpu")

                y = torch.flatten(y).to("cpu")

                pred = self.classifier(I1, I2)

                correct += accuracy_score(
                    y.detach().to(torch.float32).cpu().numpy(),
                    pred.detach().round().cpu().numpy(),
                    normalize=False,
                )

        correct /= size

        return correct


class MetaheuristicClassifierAccuracyTrainer(MetaheuristicClassifierTrainer):
    def get_task(self, classifier, dataloader):
        problem: QuantumClassifierTrainingAccuracyProblem = (
            QuantumClassifierTrainingAccuracyProblem(
                classifier=classifier, data_loader=dataloader
            )
        )

        task: Task = Task(
            problem=problem,
            max_iters=self.n_iterations,
            cutoff_value=self.cutoff,
            optimization_type=OptimizationType.MAXIMIZATION,
            enable_logging=True,
        )
        return task
