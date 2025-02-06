# %%
# Author: Manish Kumar Gupta
# Date: 04/03/2024
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection
for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018
IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
# Imports
# PyTorch
import torch
import torch.nn as nn
import torchvision.transforms as tr
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torcheval.metrics import BinaryConfusionMatrix, BinaryF1Score
from tqdm import tqdm

from src.oscd_patch_dataloader import ChangeDetectionDataset
from src.oscd_transforms import RandomFlip, RandomRot
from src.patch.config import get_config
from src.qcnn_classifier import QuantumNeuralNetwork

# GLOBAL Variables
##########################
train_loss = []
train_acc = []

val_loss = []
val_acc = []
###########################


def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    size = len(dataloader.dataset)
    num_batches: int = len(dataloader)
    metric_f1 = BinaryF1Score(threshold=0.5, device=device)
    model.train()
    correct: float = 0
    avg_loss: float = 0

    for batch, sample in enumerate(dataloader):

        I1, I2, y = sample["I1"], sample["I2"], sample["label"]
        I1 = I1.to(device)
        I2 = I2.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Compute prediction error
        pred = model(I1, I2)

        loss = loss_fn(pred, y.to(device))
        avg_loss += loss.item()

        # Backpropagation
        loss.backward()

        # Adjust learning weights
        optimizer.step()
        pred = torch.exp(pred)
        pred = torch.argmax(pred, dim=1)

        correct += accuracy_score(
            y.detach().cpu().numpy(), pred.detach().cpu().numpy(), normalize=False
        )
        metric_f1.update(pred, y)
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(y)
            print(f"Train BCE loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            print("Train BinaryF1Score: ", metric_f1.compute().item())
    train_loss.append(avg_loss / num_batches)
    train_acc.append(correct / size)
    metric_f1.reset()


def test(dataloader, model, loss_fn, device="cpu"):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    metric_f1 = BinaryF1Score(threshold=0.5, device=device)
    metric_bcm = BinaryConfusionMatrix(device=device)
    model.eval()
    avg_loss: float = 0
    correct: float = 0
    class1_correct: float = 0
    class2_correct: float = 0
    total_class2: float = 0
    class1_gt: torch.Tensor = torch.zeros(1)
    class2_gt: torch.Tensor = torch.zeros(1)

    with torch.no_grad():
        for sample in dataloader:
            I1, I2, y = sample["I1"], sample["I2"], sample["label"]
            I1 = I1.to(device)
            I2 = I2.to(device)

            y = torch.flatten(y).to(device)

            pred = model(I1, I2)
            # pred = pred.round()

            loss = loss_fn(pred, y)
            avg_loss += loss.item()

            pred = torch.exp(pred)
            pred = torch.argmax(pred, dim=1)
            # print(pred)

            correct += accuracy_score(
                y.detach().cpu().numpy(), pred.detach().cpu().numpy(), normalize=False
            )

            metric_f1.update(pred, y)
            metric_bcm.update(pred, y)

            class1_gt += y.sum()
            class2_gt += len(y) - y.sum()

            class1_correct += (
                torch.logical_and(torch.logical_not(pred), torch.logical_not(y))
                .type(torch.float)
                .sum()
                .item()
            )
            class2_correct += torch.logical_and(pred, y).type(torch.float).sum().item()

            total_class2 += y.sum().item()

    avg_loss /= num_batches

    print("Test GT change   : ", class1_gt.to("cpu").numpy())
    print("Test GT no-change   : ", class2_gt.to("cpu").numpy())
    print("class2_correct: ", class2_correct)
    print("class1_correct: ", class1_correct)
    print("size", size)

    print(
        "No Change accuracy: ",
        (100 * (class1_correct + 0.00001)) / (size - total_class2),
    )
    print(
        "Change accuracy: ",
        (100 * (class2_correct + 0.00001)) / (total_class2 + 0.00001),
    )

    correct /= size

    print("Validation BinaryF1Score: ", metric_f1.compute().item())
    print("Validation Binary Confusion Matrix: ", metric_bcm.compute())
    print(
        f"Validation Error: \n Accuracy: {(100*correct):>0.1f}%, Avg BCE loss: {avg_loss:>8f} \n"
    )
    cm = metric_bcm.compute().to("cpu").numpy()
    val_acc.append(correct)
    val_loss.append(avg_loss)
    metric_f1.reset()
    metric_bcm.reset()


def plot_results(show_results: bool = False):

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

    plt.savefig("./results/patch_train_results.png")
    print("patch_train_results.png")
    if show_results:
        plt.show()
    plt.close()


def save_test_results(dset, model, net_name, device="cpu", show_results=False):

    model.eval()

    with torch.no_grad():
        for name in dset.names:
            I1, I2, cm = dset.get_img(name)
            n1, n2 = cm.shape[0], cm.shape[1]
            X1: torch.Tensor = torch.Tensor()
            X2: torch.Tensor = torch.Tensor()
            # Generating valid patch index
            n1 = torch.floor(torch.tensor(n1 / dset.patch_side)) * dset.patch_side
            n2 = torch.floor(torch.tensor(n2 / dset.patch_side)) * dset.patch_side
            n1 = n1.to(torch.long)
            n2 = n2.to(torch.long)

            for i in range(0, n1, dset.patch_side):
                for j in range(0, n2, dset.patch_side):
                    P1 = I1[:, i : i + dset.patch_side, j : j + dset.patch_side]
                    P2 = I2[:, i : i + dset.patch_side, j : j + dset.patch_side]
                    X1 = torch.stack((X1, P1))
                    X2 = torch.stack((X2, P2))

            X1 = X1.to(device)
            X2 = X2.to(device)
            torch.cuda.empty_cache()

            pred = model(X1, X2)
            pred = torch.exp(pred)
            pred = torch.argmax(pred, dim=1)

            img = torch.reshape(
                pred,
                (
                    (n1 / dset.patch_side).to(torch.long).item(),
                    (n2 / dset.patch_side).to(torch.long).item(),
                ),
            )
            img = torch.unsqueeze(img, 0)
            img = torch.unsqueeze(img, 1)

            m = nn.Upsample(scale_factor=dset.patch_side, mode="nearest")
            img = m(img.to(torch.float))
            img = torch.squeeze(img)
            plt.imshow(img.cpu().numpy(), cmap="Greys_r", interpolation="nearest")
            if show_results:
                plt.show()
            plt.savefig(f"./results/{net_name}-{name}-gray.png")
            print(f"{net_name}-{name}-gray.png")


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)


def main():

    CONF = get_config()
    # Get cpu or gpu device for training.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    # device = "cpu"

    epochs = CONF["N_EPOCHS"]

    set_random_seed(random_seed=65822)

    start_time = datetime.now()

    if CONF["DATA_AUG"]:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None

    train_dataset = ChangeDetectionDataset(
        CONF["PATH_TO_DATASET"],
        train=True,
        patch_side=CONF["PATCH_SIDE"],
        stride=CONF["TRAIN_STRIDE"],
        transform=data_transform,
        band_type=CONF["TYPE"],
        normalise_img=CONF["NORMALISE_IMGS"],
        fp_modifier=CONF["FP_MODIFIER"],
        n_components=CONF["PCA_COMPONENTS"],
    )
    sample_weights = train_dataset.get_sample_weights()
    train_data_loader = DataLoader(
        train_dataset, batch_size=CONF["BATCH_SIZE"], shuffle=True, num_workers=1
    )

    test_dataset = ChangeDetectionDataset(
        CONF["PATH_TO_DATASET"],
        train=False,
        patch_side=CONF["PATCH_SIDE"],
        stride=CONF["TRAIN_STRIDE"],
        transform=data_transform,
        band_type=CONF["TYPE"],
        normalise_img=CONF["NORMALISE_IMGS"],
        fp_modifier=CONF["FP_MODIFIER"],
        n_components=CONF["PCA_COMPONENTS"],
    )
    test_data_loader = DataLoader(
        test_dataset, batch_size=CONF["BATCH_SIZE"], shuffle=True, num_workers=4
    )

    print("Train sample_weights: ", sample_weights)

    print("Number of QUBITS #", CONF["NUM_QUBITS"])
    model = QuantumNeuralNetwork(CONF["NUM_BANDS"], n_qubits=CONF["NUM_QUBITS"]).to(
        device
    )

    print(model)
    print(model.parameters())
    print(
        "Number of trainable parameters:",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    loss = torch.nn.NLLLoss(torch.tensor(sample_weights, device=device).float())

    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    for t in tqdm(range(epochs)):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_data_loader, model, loss, optimizer, device)
        test(test_data_loader, model, loss, device)

    # Create result folder if not exists. Needs to be called before any saving is done.
    if not os.path.exists("./results"):
        os.makedirs("./results")

    plot_results(show_results=CONF["SHOW_RESULTS"])

    torch.save(model.state_dict(), "patch_model.pth")
    print("Saved PyTorch Model State to patch_model.pth")
    print("Done!")

    print("Generating prediction images.")
    if CONF["LOAD_TRAINED"]:
        model = QuantumNeuralNetwork(CONF["NUM_BANDS"], n_qubits=CONF["NUM_QUBITS"]).to(
            device
        )
        model.load_state_dict(torch.load("./patch_model.pth"))
        print("LOAD OK")

    net_name = "patch_qclassify"
    save_test_results(
        test_dataset, model, net_name, device, show_results=CONF["SHOW_RESULTS"]
    )

    time_elapsed = datetime.now() - start_time

    print("Total execution time (hh:mm:ss.ms) {}".format(time_elapsed))


if __name__ == "__main__":
    main()
