# %%
# Author: Manish Kumar Gupta
# Date: 11/10/2022
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""
import os
import pickle
import random
import sys
import warnings
from datetime import datetime

# from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pennylane as qml
import psutil
# Imports
# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as tr
from matplotlib import pyplot as plt
from pennylane_qiskit import qiskit_session
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_machine_learning.utils import algorithm_globals
from skimage import io
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torcheval.metrics import (BinaryAccuracy, BinaryConfusionMatrix,
                               BinaryF1Score)
from tqdm import tqdm

from oscd_dataloader import ChangeDetectionDataset
from oscd_transforms import RandomFlip, RandomRot
# from quantum_manually_compiled_noisy_classifier import QuantumNeuralNetwork as QNN
from quantum_classifier import MCQuantumNeuralNetwork as QNN

# GLOBAL Variables
##########################
train_loss = []
train_acc = []

val_loss = []
val_acc = []
###########################


def test(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    metric_f1 = BinaryF1Score(threshold=0.5, device=device)
    metric_bcm = BinaryConfusionMatrix(device=device)
    model.eval()
    avg_loss, correct = 0, 0
    class1_correct, class2_correct, total_class2 = 0, 0, 0
    with torch.no_grad():  # TR: Doesn't work with new circuit...
        for i, sample in enumerate(dataloader):
            debug_log(f"\tNew sample ({i}/{num_batches})!")
            I1, I2, y = sample["I1"], sample["I2"], sample["label"]
            I1 = I1.to(device)
            I2 = I2.to(device)

            y = torch.flatten(y)

            # Qiskit circuit doesn't support batching.
            X = torch.cat((I1, I2), 1)
            X = torch.flatten(X, 1).to(device)

            # Compute prediction error
            debug_log("\tPredictions computation!")
            start_time = datetime.now()

            pred = []

            with qiskit_session(model.quantum_dev) as session:
                # TODO TR: Figure out how to do it in a batch.
                for idx, x in enumerate(X):
                    debug_log(f"\t\tSession status prior to run: {session.status()}.")
                    debug_log(f"\t\tRunning example {idx + 1}/{len(X)}...")
                    pred.append(model.forward_single(x))
                    debug_log(f"\t\tGot {pred[-1]}!")
                    debug_log(f"\t\tSession status after the run: {session.status()}.")
                    # If the session closes after the first run, check qiskit_device.py
                    # and remove the finally statement in the execute method.
                    if idx == 2:
                        break  # We're just checking if we can actually run inference with that model.
                del X  # TR: This maybe also helps with the leak.

            pred = torch.cat(pred).to(device)
            # pred = model(I1, I2)

            time_elapsed = datetime.now() - start_time
            debug_log(
                "\tPreciction computation time (hh:mm:ss.ms) {}".format(time_elapsed)
            )

            debug_log("\tLoss computation!")

            loss = loss_fn(pred, y.to(torch.float32).to(device))
            avg_loss += loss.item()

            debug_log("\tAccuracy computation start!")
            correct += accuracy_score(
                y.detach().to(torch.float32).cpu().numpy(),
                pred.detach().round().cpu().numpy(),
                normalize=False,
            )

            class1_correct += (
                ((pred >= 0.5).type(torch.float) == 0.0).type(torch.float).sum().item()
            )
            class2_correct += (
                ((pred >= 0.5).type(torch.float) == 1.0).type(torch.float).sum().item()
            )
            total_class2 += y.sum().item()

            metric_f1.update(pred, y.to(torch.float32))
            metric_bcm.update(pred, y.type(torch.int32))

            debug_log(
                f"\t\tGPU memory leak check:{torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()}"
            )
            debug_log(f"\t\tRAM leak check: {psutil.virtual_memory().percent}")

    avg_loss /= num_batches
    correct /= size
    print("Validation BinaryF1Score: ", metric_f1.compute().item())
    print("Validation Binary Confusion Matrix: ", metric_bcm.compute())
    print(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg BCE loss: {avg_loss:>8f} \n"
    )
    val_acc.append(correct)
    val_loss.append(avg_loss)
    metric_f1.reset()
    metric_bcm.reset()


def plot_results():

    epochs = range(1, len(train_loss) + 1)

    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4)
    ax1 = fig.add_subplot(2, 1, 1)  # two rows, one column, first plot
    ax1.set_title("Training accuraccy and test accuracy")
    ax1.set_ylabel("accuracy")
    ax1.set_xlabel("epochs")
    ax1.set_ylim(0, 1)
    ax1.plot(epochs, train_acc, "ko-", label="Training acc")
    ax1.plot(epochs, val_acc, "tab:orange", label="Test acc")
    ax1.legend()

    ax2 = fig.add_subplot(2, 1, 2)  # two rows, one column, first plot
    ax2.set_title("Training loss and test loss")
    ax2.set_ylabel("BCE loss")
    ax2.set_xlabel("epochs")
    ax2.set_ylim(0, 1)
    ax2.plot(epochs, train_loss, "ko-", label="Training loss")
    ax2.plot(epochs, val_loss, "tab:orange", label="Test loss")
    ax2.legend()
    plt.savefig("./results/train_results.png")
    print("train_results.png")
    plt.show()
    plt.close()


def save_test_results(dset, model, net_name, device="cpu"):

    model.eval()

    with torch.no_grad():
        for name in dset.names:
            I1, I2, cm = dset.get_img(name)
            n1, n2 = cm.shape[0], cm.shape[1]
            # print('name', name)
            X1 = []
            X2 = []
            for i in range(n1):
                for j in range(n2):
                    P1 = I1[:, i, j]
                    P2 = I2[:, i, j]
                    X1.append(P1)
                    X2.append(P2)

            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            X1 = X1.to(device)
            X2 = X2.to(device)
            torch.cuda.empty_cache()

            # print(X1)
            pred = model(X1, X2)
            pred = torch.round(pred)
            plt.imshow(
                torch.reshape(pred, (n1, n2)).cpu().numpy(),
                cmap="Greys_r",
                interpolation="nearest",
            )
            plt.show()
            plt.savefig(f"./results/{net_name}-{name}-gray.png")
            print(f"{net_name}-{name}-gray.png")

            I = np.stack(
                (
                    1 * 225 * cm,
                    225 * torch.reshape(pred, (n1, n2)).cpu().numpy(),
                    1 * 255 * cm,
                ),
                2,
            )

            # TODO TR: Why isn't I used? Why is it computed?


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)

    algorithm_globals.random_seed = random_seed


def debug_log(string, should_log=True):
    if should_log:
        print(string + f" ({datetime.now()})")


def main():

    ################### Settings ############################
    FP_MODIFIER = 1  # Tuning parameter, use 1 if unsure

    BATCH_SIZE = 500  # 32
    PATCH_SIDE = 1

    NORMALISE_IMGS = True

    TRAIN_STRIDE = 10  # int(PATCH_SIDE/2) - 1

    TYPE = 2  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands

    NUM_BANDS = TYPE + 3

    PCA_COMPONENTS = 4  # 4

    # NUM_QUBITS = 2 * PATCH_SIDE*PATCH_SIDE * NUM_BANDS if NUM_BANDS <= 4 else  2 * PATCH_SIDE*PATCH_SIDE * PCA_COMPONENTS
    NUM_QUBITS = (
        1 * PATCH_SIDE * PATCH_SIDE * NUM_BANDS
        if NUM_BANDS <= 4
        else 1 * PATCH_SIDE * PATCH_SIDE * PCA_COMPONENTS
    )

    DATA_AUG = False

    NUM_WORKERS = 1
    #########################################################

    PATH_TO_DATASET = os.environ["SIPWQNN_DATA_PATH"]

    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # device = "cpu"

    set_random_seed(random_seed=65822)

    start_time = datetime.now()

    if DATA_AUG:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None

    test_dataset = ChangeDetectionDataset(
        PATH_TO_DATASET,
        train=False,
        patch_side=PATCH_SIDE,
        stride=TRAIN_STRIDE,
        transform=data_transform,
        band_type=TYPE,
        normalise_img=NORMALISE_IMGS,
        fp_modifier=FP_MODIFIER,
        n_components=PCA_COMPONENTS,
    )
    sample_weights = test_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS
    )

    print("Getting remote device...")
    backend_name = "ibm_brisbane"
    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    dev = qml.device(
        "qiskit.remote", wires=backend.num_qubits, backend=backend
    )  #  WORKS!
    # dev = qml.device("default.qubit")  # For tests

    model = QNN(n_qubits=NUM_QUBITS, quantum_dev=dev, torch_device=device)

    initial_weights = model.parameters_list()

    model.load_state_dict(
        torch.load(
            "./refac_mc_noiseless_default.pth", map_location=torch.device(device)
        )
    )
    print("LOAD OK")

    loaded_weights = model.parameters_list()

    print("Changing device to a real one.")

    # model.set_quantum_layer(NUM_QUBITS, dev)
    # model.quantum_layer.weights = quantum_layer_weights

    # print(model.quantum_layer.qnode)
    # print(model.quantum_layer.weights)
    # return

    # print('CUDA memory allocation after model:', torch.cuda.memory_allocated(device))

    print(model)
    print(model.parameters())
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    loss = nn.BCELoss()
    start_time = datetime.now()
    test(test_data_loader, model, loss, device)
    time_elapsed = datetime.now() - start_time
    debug_log("\tTest time (hh:mm:ss.ms) {}".format(time_elapsed))

    # plot_results()

    return

    torch.save(model.state_dict(), "mc_noiseless_default.pth")
    print("Saved PyTorch Model State to model.pth")
    print("Done!")

    print(f"\t\tTrain acc {train_acc}")
    print(f"\t\tTrain loss {train_loss}")
    print(f"\t\tTest acc {val_acc}")
    print(f"\t\tTest loss {val_loss}")

    return

    print("Generating prediction images.")
    LOAD_TRAINED = False
    if LOAD_TRAINED:
        model = QNN(n_qubits=NUM_QUBITS).to(device)
        model.load_state_dict(torch.load("./model.pth"))
        print("LOAD OK")

    net_name = "qclassify"
    save_test_results(test_dataset, model, net_name, device)

    time_elapsed = datetime.now() - start_time
    print("Total execution time (hh:mm:ss.ms) {}".format(time_elapsed))


if __name__ == "__main__":
    main()
