import pickle as pkl
import random
from datetime import datetime
from typing import List, Sequence

import numpy as np
import torch
from qiskit_machine_learning.utils import algorithm_globals
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from oscd_dataloader import \
    ChangeDetectionDataset  # TODO: Fix this import. Python is dumb.


def get_data_lengths(batch_nums: Sequence[int]) -> List[int]:
    lengths: List[int] = []

    for n in batch_nums:
        with open(f"Auxilliary/sipqnn.preds.batch{n}.bkp.txt", "r") as f:
            lengths.append(len(f.readlines()))

    return lengths


def prepare_analysis_data() -> None:

    batch_nums: List[int] = [0, 1, 29, 61]

    data_lengths: List[int] = get_data_lengths(batch_nums)

    brisbane_data = []

    with open("Auxilliary/brisbane_data.pkl", "rb") as f:
        brisbane_data = pkl.load(f)

    assert sum(data_lengths) == len(brisbane_data)

    sim_data = []

    for i, n in enumerate(batch_nums):
        with open(f"Auxilliary/true_model.sim.batch{n}.pkl", "rb") as f:
            # with open(f"Auxilliary/sim.batch{n}.pkl", "rb") as f:
            batch_data = pkl.load(f)

        for j in range(data_lengths[i]):
            sim_data.append(batch_data[j])

    assert len(sim_data) == len(brisbane_data)

    with open("t_final_analysis_data.pkl", "wb") as f:
        pkl.dump({"real": brisbane_data, "sim": sim_data}, f)


def process_pcss_data() -> None:

    data_files = [f"Auxilliary/sipqnn.preds.batch{i}.bkp.txt" for i in [0, 1, 29, 61]]
    processed_data = []

    for file in data_files:
        with open(file, "r") as f:
            for l in f.readlines():
                l = l.strip()
                l = l.replace("tensor", "")
                l = l.replace("([", "")
                l = l.replace("])", "")
                input_pred = l.split(" -> ")

                X = [float(x) for x in input_pred[0].split(", ")]
                pred = [float(input_pred[1])]

                processed_data.append((Tensor(X), Tensor(pred)))

    with open("Auxilliary/brisbane_data.pkl", "wb") as f:
        pkl.dump(processed_data, f)

    print(f"\tBrisbane data ({len(processed_data)} points) processed.")


if __name__ == "__main__":
    print("Processing start.")
    # process_pcss_data()
    prepare_analysis_data()
    print("Done.")
