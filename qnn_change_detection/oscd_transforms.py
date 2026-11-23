# Author: Manish Kumar Gupta
# Date: 11/10/2022
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""
import random

import numpy as np
import torch


class RandomFlip(object):
    """Flip randomly the images in a sample."""

    """ TODO: Need to implement transforms!!  """

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample["I1"], sample["I2"], sample["label"]

        #        if random.random() > 0.5:
        #            I1 =  I1.numpy()[:,:,::-1].copy()
        #            I1 = torch.from_numpy(I1)
        #            I2 =  I2.numpy()[:,:,::-1].copy()
        #            I2 = torch.from_numpy(I2)
        #            label =  label.numpy()[:,::-1].copy()
        #            label = torch.from_numpy(label)

        return {"I1": I1, "I2": I2, "label": label}


class RandomRot(object):
    """Rotate randomly the images in a sample."""

    """ TODO: Need to implement transforms!!  """

    #     def __init__(self):
    #         return

    def __call__(self, sample):
        I1, I2, label = sample["I1"], sample["I2"], sample["label"]

        #        n = random.randint(0, 3)
        #        if n:
        #            I1 =  sample['I1'].numpy()
        #            I1 = np.rot90(I1, n, axes=(1, 2)).copy()
        #            I1 = torch.from_numpy(I1)
        #            I2 =  sample['I2'].numpy()
        #            I2 = np.rot90(I2, n, axes=(1, 2)).copy()
        #            I2 = torch.from_numpy(I2)
        #            label =  sample['label'].numpy()
        #            label = np.rot90(label, n, axes=(0, 1)).copy()
        #            label = torch.from_numpy(label)

        return {"I1": I1, "I2": I2, "label": label}
