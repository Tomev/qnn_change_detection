import os

import matplotlib.pyplot as plt
import torch

from src.mlflow_utils import get_model_state_dict, get_run_config
from src.pixel_level.utils import get_datasets
from src.quantum_classifier import QuantumNeuralNetwork


def save_test_results(dset, model, run_id, device="cpu", show_results=False):

    model.eval()

    with torch.no_grad():
        for name in dset.names:
            I1, I2, cm = dset.get_img(name)
            n1, n2 = cm.shape[0], cm.shape[1]
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

            pred = model(X1, X2)
            pred = torch.round(pred)

            plt.imshow(
                torch.reshape(pred, (n1, n2)).cpu().numpy(),
                cmap="Greys_r",
                interpolation="nearest",
            )

            if show_results:
                plt.show()

            plt.savefig(f"./results/{run_id}-{name}-gray.png")
            print(f"{run_id}-{name}-gray.png")


def generate_prediction_images(run_id: str) -> None:
    # Remember to ensure mlflow uri is set prior to calling this function.

    # Create result folder if not exists. Needs to be called before any saving is done.
    if not os.path.exists("./results"):
        os.makedirs("./results")

    exp_conf = get_run_config(run_id)

    print("Generating prediction images.")
    model = QuantumNeuralNetwork(
        n_qubits=exp_conf["NUM_QUBITS"], quantum_dev=exp_conf["general"]["DEV"]
    ).to(exp_conf["general"]["TORCH_DEVICE"])

    # TODO TR: Needs testing.
    model.load_state_dict(get_model_state_dict(run_id))
    print("LOAD OK")

    _, test_dataset = get_datasets(exp_conf["general"])

    save_test_results(test_dataset, model, run_id, exp_conf["general"]["TORCH_DEVICE"])


def main():
    run_ids = []

    for run_id in run_ids:
        generate_prediction_images(run_id)


if __name__ == "__main__":
    main()
