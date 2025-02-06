# %%
# Author: Manish Kumar Gupta
# Date: 11/10/2022
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection
for multispectral earth observation using convolutional neural networks. In IGARSS
2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""
import random
from datetime import datetime

import mlflow
import numpy as np
import pennylane as qml
# Imports
import torch
import torch.nn as nn
from mlflow.models import infer_signature
from niapy.callbacks import Callback
from sklearn.metrics import accuracy_score
from torcheval.metrics import BinaryConfusionMatrix, BinaryF1Score
from tqdm import tqdm

from src.metaheuristics.trainer import MetaheuristicClassifierAccuracyTrainer
from src.mlflow_utils import get_exp_id, save_exp_conf
from src.pixel_level.config import prepare_experiments_configs
from src.pixel_level.utils import prepare_data_loaders, set_random_seed
from src.quantum_classifier import MCQuantumNeuralNetwork as QNN


def generate_training_callback(prefix: str) -> Callback:
    cb = Callback()

    def cb_training_after(population, fitness, best_x, best_fitness, **params):
        print(f"\tTraining accuracy: {best_fitness}")
        metric_label = prefix + "meta_train_acc"

        mlflow.log_metric(
            metric_label, best_fitness, timestamp=int(datetime.now().timestamp())
        )

    cb.after_iteration = cb_training_after

    return cb


def prepare_metaheuristic_trainer_callbacks(initialization=False, tuning=False):
    prefix = ""

    if initialization:
        prefix = "init_"
    if tuning:
        prefix = "tuning_"

    callbacks = [generate_training_callback(prefix)]

    return callbacks


def prepare_metaheuristic_trainer(
    conf, init_function=None, initialization=False, tuning=False
):
    trainer_config = conf["metaheuristic"]["training"]
    if initialization:
        trainer_config = conf["metaheuristic"]["initialization"]
    if tuning:
        trainer_config = conf["metaheuristic"]["tuning"]

    algorithm = trainer_config["algorithm"](
        population_size=trainer_config["population_size"],
        initializaiton_function=init_function,
        max_velocity=trainer_config["particle_speed"],
        seed=conf["general"]["SEED"],
        callbacks=prepare_metaheuristic_trainer_callbacks(initialization),
    )

    # TODO TR: Ensure this is still needed.
    if init_function:
        algorithm.initialization_function = init_function

    trainer: MetaheuristicClassifierAccuracyTrainer = (
        MetaheuristicClassifierAccuracyTrainer(
            algorithm=algorithm,
            n_iterations=trainer_config["generations_number"],
            cutoff=trainer_config["cutoff"],
        )
    )

    return trainer


def metaheuristic_training(
    data, model, exp_conf, initialization=False, init_function=None, tuning=False
):

    trainer = prepare_metaheuristic_trainer(
        exp_conf,
        initialization=initialization,
        init_function=init_function,
        tuning=tuning,
    )

    trainer.fit(classifier=model, data_loader=data)


def get_preds(model, sample, device="cpu"):
    I1, I2 = sample["I1"], sample["I2"]
    I1 = I1.to(device)
    I2 = I2.to(device)

    X = torch.cat((I1, I2), 1)
    X = torch.flatten(X, 1).to(device)
    X.requires_grad = False

    pred_list = []
    for x in X:
        print(x)
        pred_list.append(model.forward_single(x))

    return torch.cat(pred_list).to(device)


def compute_accuracy(pred, y):
    return accuracy_score(
        y.detach().to(torch.float32).cpu().numpy(),
        pred.detach().round().cpu().numpy(),
        normalize=False,
    )


def epoch_train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)

    model.train()
    correct, avg_loss = 0, 0
    for batch, sample in enumerate(dataloader):

        y = torch.flatten(sample["label"])

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        pred = get_preds(model, sample, device=device)

        loss = loss_fn(pred, y.to(torch.float32).to(device))
        avg_loss += loss.item()

        loss.backward()
        optimizer.step()

        correct += compute_accuracy(pred, y)

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"Train BCE loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss = avg_loss / len(dataloader)
    train_acc = correct / size

    return train_acc, train_loss


def test(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)

    metric_f1 = BinaryF1Score(threshold=0.5, device=device)
    metric_bcm = BinaryConfusionMatrix(device=device)

    model.eval()
    avg_loss, correct = 0, 0
    with torch.no_grad():
        for sample in dataloader:

            y = torch.flatten(sample["label"]).to(device)
            pred = get_preds(model, sample, device=device)

            loss = loss_fn(pred, y.to(torch.float32))
            avg_loss += loss.item()

            correct += compute_accuracy(pred, y)

            metric_f1.update(pred, y.to(torch.float32))
            metric_bcm.update(pred, y.type(torch.int32))

    val_acc = correct / size
    val_loss = avg_loss / len(dataloader)
    print("Validation Binary F1Score: ", metric_f1.compute().item())
    print("Validation Binary Confusion Matrix: ", metric_bcm.compute())
    print(
        f"Validation Error: \n Accuracy: {(100*val_acc):>0.1f}%, Avg BCE loss: {val_loss:>8f} \n"
    )

    metric_f1.reset()
    metric_bcm.reset()

    return val_acc, val_loss


def experiment(exp_conf):

    set_random_seed(random_seed=exp_conf["general"]["SEED"])
    print(f"Using {exp_conf['general']['TORCH_DEVICE']} torch device")

    tracker = qml.Tracker(exp_conf["general"]["DEV"])
    tracker.active = True

    print(f"Initial tracker data: {tracker.totals}")

    train_data_loader, test_data_loader = prepare_data_loaders(exp_conf["general"])

    start_time = datetime.now()

    model = QNN(
        n_qubits=exp_conf["general"]["NUM_QUBITS"],
        quantum_dev=exp_conf["general"]["DEV"],
        torch_device=exp_conf["general"]["TORCH_DEVICE"],
    ).to(exp_conf["general"]["TORCH_DEVICE"])

    loss = nn.BCELoss()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    if exp_conf["metaheuristic"]["initialization"]:
        metaheuristic_training(train_data_loader, model, exp_conf, initialization=True)

    val_acc, val_loss = test(
        test_data_loader, model, loss, exp_conf["general"]["TORCH_DEVICE"]
    )
    now = int(datetime.now().timestamp())
    mlflow.log_metric("after_init_val_loss", val_loss, timestamp=now)
    mlflow.log_metric("after_init_val_acc", val_acc, timestamp=now)

    if exp_conf["metaheuristic"]["training"]:

        # In case tuning was done
        initial_weights = model.parameters_list()

        def init_function(task, population_size, rng, **_kwargs):

            multiplier_mod: float = 0.1
            pop = [np.array(initial_weights)]
            fpop = [task.eval(pop[-1])]

            while len(pop) < population_size:
                pop.append(
                    np.array(
                        [
                            v * (1 + random.uniform(-multiplier_mod, multiplier_mod))
                            for v in initial_weights
                        ]
                    )
                )
                fpop.append(task.eval(pop[-1]))

            return pop, fpop

        metaheuristic_training(
            train_data_loader, model, exp_conf, init_function=init_function
        )
    else:
        for t in tqdm(
            range(exp_conf["general"]["N_EPOCHS"]), desc="Gradient training epochs\n"
        ):
            train_acc, train_loss = epoch_train(
                train_data_loader,
                model,
                loss,
                optimizer,
                exp_conf["general"]["TORCH_DEVICE"],
            )
            now = int(datetime.now().timestamp())
            mlflow.log_metric("train_loss", train_loss, step=t + 1, timestamp=now)
            mlflow.log_metric("train_acc", train_acc, step=t + 1, timestamp=now)
            val_acc, val_loss = test(
                test_data_loader, model, loss, exp_conf["general"]["TORCH_DEVICE"]
            )
            now = int(datetime.now().timestamp())
            mlflow.log_metric("val_loss", val_loss, step=t + 1, timestamp=now)
            mlflow.log_metric("val_acc", val_acc, step=t + 1, timestamp=now)

    if exp_conf["metaheuristic"]["tuning"]:

        initial_weights = model.parameters_list()

        def init_function(task, population_size, rng, **_kwargs):

            multiplier_mod: float = 0.1
            pop = [np.array(initial_weights)]
            fpop = [task.eval(pop[-1])]

            while len(pop) < population_size:
                pop.append(
                    np.array(
                        [
                            v * (1 + random.uniform(-multiplier_mod, multiplier_mod))
                            for v in initial_weights
                        ]
                    )
                )
                fpop.append(task.eval(pop[-1]))

            return pop, fpop

        metaheuristic_training(
            train_data_loader, model, exp_conf, init_function=init_function
        )

    val_acc, val_loss = test(
        test_data_loader, model, loss, exp_conf["general"]["TORCH_DEVICE"]
    )
    now = int(datetime.now().timestamp())
    mlflow.log_metric("final_val_loss", val_loss, timestamp=now)
    mlflow.log_metric("final_val_acc", val_acc, timestamp=now)

    print(f"After training tracker data:\n{tracker.totals}")

    time_elapsed = datetime.now() - start_time

    print("Total training time (hh:mm:ss.ms) {}".format(time_elapsed))

    mlflow.log_dict(model.state_dict(), "model_state_dict")

    # TODO TR: Figure out how to save the schema properly.
    # Especially for Manish code with double input.
    mlflow.pytorch.log_model(model, "model")

    print("Saved PyTorch Model State as mlflow artifact.")
    print("Done!")


def main() -> None:
    print("Starting experiments.")
    # Also try: https://mlflow.org/docs/latest/tracking/tracking-api.html#parallel-runs

    mlflow.set_tracking_uri("http://127.0.0.1:8080")

    for exp_conf in tqdm(
        prepare_experiments_configs(), desc="QML_MC_Training_Experiments"
    ):
        experiment_id = get_exp_id(exp_conf, "pixel-level")

        mlflow.start_run(experiment_id=experiment_id)

        save_exp_conf(exp_conf)

        experiment(exp_conf)

        mlflow.end_run()

    print("Experiments finished.")


if __name__ == "__main__":
    main()
