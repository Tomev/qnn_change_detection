import random

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.oscd_dataloader import ChangeDetectionDataset


def set_random_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(random_seed)
    random.seed(random_seed)


def get_datasets(CONF):
    # Returns a tuple of datasets. First for training second for tests.
    return (
        ChangeDetectionDataset(
            CONF["PATH_TO_DATASET"],
            train=train,
            patch_side=CONF["PATCH_SIDE"],
            stride=CONF["TRAIN_STRIDE"],
            transform=CONF["DATA_TRANSFORM"],
            band_type=CONF["TYPE"],
            normalise_img=CONF["NORMALISE_IMGS"],
            fp_modifier=CONF["FP_MODIFIER"],
            n_components=CONF["PCA_COMPONENTS"],
        )
        for train in [True, False]
    )


def prepare_data_loader(dataset, CONF):
    sample_weights = dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    dataloader = DataLoader(
        dataset, batch_size=CONF["BATCH_SIZE"], sampler=sampler, num_workers=4
    )

    return dataloader


def prepare_data_loaders(CONF):

    train_dataset, test_dataset = get_datasets(CONF)
    train_data_loader = prepare_data_loader(train_dataset, CONF)
    test_data_loader = prepare_data_loader(test_dataset, CONF)

    return train_data_loader, test_data_loader
