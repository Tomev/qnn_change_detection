import math
import pickle as pkl
from typing import Tuple

import torch
from numpy import std
from scipy.special import logit
from scipy.stats import chisquare
from sklearn.metrics import accuracy_score
from torch import Tensor
from torcheval.metrics import (BinaryAccuracy, BinaryConfusionMatrix,
                               BinaryF1Score)


def get_analysis_data() -> Tuple[Tensor, Tensor, Tensor]:
    sim = []
    real = []
    labels = []

    with open("Auxilliary/final_analysis_data.pkl", "rb") as f:
        data = pkl.load(f)

    n_samples: int = len(data["sim"])

    for i in range(n_samples):
        sim.append(data["sim"][i][1])
        real.append(data["real"][i][1])
        labels.append(int(not data["sim"][i][2]))  # not -> bitflip
        # labels.append(int(data["sim"][i][2]))

    return Tensor(sim), Tensor(real), Tensor(labels)


def get_class_correct(pred: Tensor, y: Tensor) -> Tuple[int, int]:

    n_samples: int = len(y)
    n_c0: int = 0
    n_c1: int = 0

    for i in range(n_samples):

        if pred[i] >= 0.5 and y[i] == 1:
            n_c1 += 1

        if pred[i] < 0.5 and y[i] == 0:
            n_c0 += 1

    return n_c0, n_c1


def sureness(data: Tensor) -> float:
    return float(2 * sum([abs(d - 0.5) for d in data]) / len(data))


def confidence(data: Tensor, gts: Tensor) -> float:

    for d in data:
        assert d > 0
        assert d < 1

    l = [abs(data[i] - gts[i]) for i in range(len(data))]
    print(std(l))
    return 1 - (sum(l) / len(data))


def main() -> None:
    sim, real, labels = get_analysis_data()
    n_samples = len(sim)

    print(torch.std(sim))
    print(torch.std(real))

    metric_f1 = BinaryF1Score(threshold=0.5)
    metric_bcm = BinaryConfusionMatrix()

    print(sum(labels))

    print("\n\n")

    metric_f1.reset()
    metric_f1.update(real, labels)
    f1 = metric_f1.compute().item()
    acc = accuracy_score(
        labels,
        real.round(),
        normalize=True,
    )
    sure = sureness(real)
    conf = confidence(real, labels)
    n_c0, n_c1 = get_class_correct(real, labels)

    print(f"\treal")
    print(f"\t\tf1: {f1}")
    print(f"\t\tacc: {acc}")
    print(f"\t\tsur: {sure}")
    print(f"\t\tconfidence: {conf}")
    print(f"\t\tn_c0={n_c0}, n_c1={n_c1}, imbalance={n_c0 - n_c1}")
    print("\n\n")

    # labels = torch.Tensor([(int(l) + 1) % 2 for l in labels])

    metric_f1.reset()
    metric_f1.update(sim, labels)
    f1 = metric_f1.compute().item()
    acc = accuracy_score(
        labels,
        sim.round(),
        normalize=True,
    )
    sure = sureness(sim)
    conf = confidence(sim, labels)
    n_c0, n_c1 = get_class_correct(sim, labels)
    print(f"\tsim:")
    print(f"\t\tf1: {f1}")
    print(f"\t\tacc: {acc}")
    print(f"\t\tsur: {sure}")
    print(f"\t\tconfidence: {conf}")
    print(f"\t\tn_c0={n_c0}, n_c1={n_c1}, imbalance={n_c0 - n_c1}")

    print("\n\n")

    n_r_c1_close = 0
    n_r_c0_close = 0

    for i in range(n_samples):
        if labels[i] == 1 and 0.50 <= real[i] <= 0.55:
            n_r_c1_close += 1

        if labels[i] == 0 and 0.45 <= real[i] < 0.5:
            n_r_c0_close += 1

    print(n_r_c0_close)
    print(n_r_c1_close)

    return

    rss = [int(1024 * logit(s)) for s in sim]
    rsr = [int(1024 * logit(r)) for r in real]

    for i in range(len(rss)):
        if rss[i] == 0:
            rss[i] += 1

    print(chisquare(rsr, rss))

    # print(sim)


if __name__ == "__main__":
    print("Analysis start.")
    main()
    print("Analysis done.")
