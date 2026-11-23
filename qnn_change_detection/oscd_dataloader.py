# Author: Manish Kumar Gupta
# Date: 11/10/2022
# Project Info
"""
Change detection using Quantum neural network model.

Data used from the OSCD dataset:
Daudt, R.C., Le Saux, B., Boulch, A. and Gousseau, Y., 2018, July. Urban change detection for multispectral earth observation using convolutional neural networks. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 2115-2118). IEEE.

"""
import os
import sys
from math import ceil, exp, floor, sqrt

import matplotlib.pyplot as plt
import numpy as np
import torch
from pandas import read_csv
from scipy.ndimage import zoom
from skimage import io
from sklearn.decomposition import PCA
from sklearn.feature_extraction import image
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class ChangeDetectionDataset(Dataset):
    """Change Detection dataset class, used for both training and test data."""

    def __init__(
        self,
        path,
        train=True,
        patch_side=96,
        stride=None,
        use_all_bands=False,
        transform=None,
        band_type=0,
        normalise_img=True,
        fp_modifier=1,
        n_components=4,
    ):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        # basics
        self.transform = transform
        self.path = path
        self.patch_side = patch_side
        self.band_type = band_type
        self.normalise_img = normalise_img
        self.fp_modifier = fp_modifier
        self.std_scaler1 = StandardScaler()
        self.std_scaler2 = StandardScaler()
        self.pca1 = PCA(n_components=n_components)
        self.pca2 = PCA(n_components=n_components)

        if not stride:
            self.stride = 1
        else:
            self.stride = stride

        if train:
            fname = "train.txt"
            sub_path = "Onera Satellite Change Detection dataset - Train Labels"
        else:
            fname = "test.txt"
            sub_path = "Onera Satellite Change Detection dataset - Test Labels"

        # print(os.path.join(path, 'Onera Satellite Change Detection dataset - Images',fname))
        self.names = read_csv(
            os.path.join(
                path, "Onera Satellite Change Detection dataset - Images", fname
            )
        ).columns
        self.n_imgs = self.names.shape[0]
        # print(self.names)

        n_pix = 0
        true_pix = 0

        # load images
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        for im_name in tqdm(self.names):
            # load and store each image
            # print(os.path.join(self.path, sub_path, im_name))
            I1, I2, cm = read_sentinel_img_trio(
                os.path.join(self.path, sub_path),
                im_name,
                self.band_type,
                self.normalise_img,
            )

            if train == True and (band_type == 2 or band_type == 3):
                I = I1.reshape((I1.shape[0] * I1.shape[1]), I1.shape[2])
                I = self.std_scaler1.fit_transform(I)
                I1 = self.pca1.fit_transform(I).reshape(
                    I1.shape[0], I1.shape[1], n_components
                )
                # print('singular_values_I1: ', self.pca1.singular_values_)

                I = I2.reshape((I2.shape[0] * I2.shape[1]), I2.shape[2])
                I = self.std_scaler2.fit_transform(I)
                I2 = self.pca2.fit_transform(I).reshape(
                    I2.shape[0], I2.shape[1], n_components
                )
                # print('singular_values_I2: ', self.pca2.singular_values_)
                # print('train:', I1.shape)

            elif train == False and (band_type == 2 or band_type == 3):
                I = I1.reshape((I1.shape[0] * I1.shape[1]), I1.shape[2])
                I = self.std_scaler1.fit_transform(I)
                I1 = self.pca1.fit_transform(I).reshape(
                    I1.shape[0], I1.shape[1], n_components
                )
                # print('singular_values_I1: ', self.pca1.singular_values_)

                I = I2.reshape((I2.shape[0] * I2.shape[1]), I2.shape[2])
                I = self.std_scaler2.fit_transform(I)
                I2 = self.pca2.fit_transform(I).reshape(
                    I2.shape[0], I2.shape[1], n_components
                )
                # print('singular_values_I2: ', self.pca2.singular_values_)
                # print('test:', I1.shape)

            # print(I1.shape)
            self.imgs_1[im_name] = reshape_for_torch(I1)
            self.imgs_2[im_name] = reshape_for_torch(I2)
            self.change_maps[im_name] = cm

            # print(self.imgs_1[im_name].shape)

            flag_p = 0
            if flag_p == 1:
                print("Image1 shape: {}".format(I1.shape))
                print("Image2 shape: {}".format(I2.shape))
                print("Image cm shape: {}".format(cm.shape))
                patches = image.extract_patches_2d(I1, (1, 1))
                print("Patches shape: {}".format(patches.shape))
                print(patches[1])
                print(patches[1][:, :, 0])
                flag_p = 0

            # print(cm.shape)
            s = cm.shape
            n_pix += np.prod(s)
            true_pix += cm.sum()  # true_pix is number of pixel that is True or changed.
            # print(n_pix, true_pix)
            # print(cm.dtype)

            # calculate the number of patches
            s = self.imgs_1[im_name].shape
            # print(s)
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            # print(self.patch_side, self.stride)
            # print(n1,n2)
            # sys.exit(-1)
            n_patches_i = n1 * n2
            self.n_patches_per_image[im_name] = n_patches_i
            self.n_patches += n_patches_i
            # print('num_patch:',self.n_patches)
            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (
                        im_name,
                        [
                            self.stride * i,
                            self.stride * i + self.patch_side,
                            self.stride * j,
                            self.stride * j + self.patch_side,
                        ],
                        [self.stride * (i + 1), self.stride * (j + 1)],
                    )
                    # print(current_patch_coords)
                    self.patch_coords.append(current_patch_coords)
            # sys.exit(-1)

        self.weights = [
            self.fp_modifier * 2 * true_pix / n_pix,
            2 * (n_pix - true_pix) / n_pix,
        ]  # What is this?
        self.class_weights = [1 / true_pix, 1 / (n_pix - true_pix)]
        # self.class_weights = [true_pix/n_pix, (n_pix - true_pix)/n_pix]
        # print('weights:', self.class_weights)
        # print(self.imgs_1[im_name].shape)

    def get_sample_weights(self):
        self.sample_weights = []
        weight = 0
        for current_patch in self.patch_coords:
            current_patch_coords = current_patch
            im_name = current_patch_coords[0]
            # print('im_name', im_name)
            limits = current_patch_coords[1]
            centre = current_patch_coords[2]
            label = self.change_maps[im_name][
                limits[0] : limits[1], limits[2] : limits[3]
            ]
            weight = self.class_weights[0] if label == True else self.class_weights[1]
            # print('weight: ', weight)
            self.sample_weights.append(weight)

        return self.sample_weights

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches
        # return 5

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]

        # print(limits[0],limits[1],limits[2],limits[3])
        # print(self.imgs_1[im_name][0:3,:,:])
        I1 = self.imgs_1[im_name][:, limits[0] : limits[1], limits[2] : limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0] : limits[1], limits[2] : limits[3]]

        # pltsize=1
        # plt.figure(figsize=(10*pltsize, pltsize*1.1))
        # print(I1)
        # plt.imshow(I1[-1,:,:])
        # print(I2)
        # plt.imshow(I2[-1,:,:])

        label = self.change_maps[im_name][limits[0] : limits[1], limits[2] : limits[3]]
        # if label == True:
        #     print(label)

        label = torch.from_numpy(1 * np.array(label)).to(torch.bool)

        sample = {"I1": I1, "I2": I2, "label": label}
        # sample = torch.cat((I1,I2),1)
        # sample = torch.flatten(sample)

        if self.transform:
            sample = self.transform(sample)

        # print(sample)
        # print(label[:10])
        # return sample, label
        return sample


def read_sentinel_img_trio(path, im_name, band_type, normalise_img):
    """Read cropped Sentinel-2 image pair and change map."""
    #     read images
    type = band_type
    # print(os.path.join(path , './../Onera Satellite Change Detection dataset - Images', im_name,'imgs_1/'))
    if type == 0:
        I1 = read_sentinel_img(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_1/",
            ),
            normalise_img,
        )
        I2 = read_sentinel_img(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_2/",
            ),
            normalise_img,
        )
    elif type == 1:
        I1 = read_sentinel_img_4(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_1/",
            ),
            normalise_img,
        )
        I2 = read_sentinel_img_4(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_2/",
            ),
            normalise_img,
        )
    elif type == 2:
        I1 = read_sentinel_img_leq20(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_1/",
            ),
            normalise_img,
        )
        I2 = read_sentinel_img_leq20(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_2/",
            ),
            normalise_img,
        )
    elif type == 3:
        I1 = read_sentinel_img_leq60(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_1/",
            ),
            normalise_img,
        )
        I2 = read_sentinel_img_leq60(
            os.path.join(
                path,
                "./../Onera Satellite Change Detection dataset - Images",
                im_name,
                "imgs_2/",
            ),
            normalise_img,
        )

    cm = io.imread(os.path.join(path, im_name, "cm/cm.png"), as_gray=True) != 0

    # crop if necessary
    s1 = I1.shape  # It seems that it is a patch which might not properly in every case.
    s2 = I2.shape
    I2 = np.pad(
        I2, ((0, s1[0] - s2[0]), (0, s1[1] - s2[1]), (0, 0)), "edge"
    )  # Can lead to negative index and hence error!!

    return I1, I2, cm


def read_sentinel_img(path, normalise_img):
    """Read cropped Sentinel-2 image: RGB bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")

    I = np.stack((r, g, b), axis=2).astype("float32")

    if normalise_img:
        I = (I - I.mean()) / I.std()

    # print(I)
    # plot the pixel values
    # plt.hist(I.ravel(), bins=100, density=True)
    # plt.xlabel("pixel values")
    # plt.ylabel("relative frequency")
    # plt.title("distribution of pixels")

    return I


def read_sentinel_img_4(path, normalise_img):
    """Read cropped Sentinel-2 image: RGB and NIR bands."""
    im_name = os.listdir(path)[0][:-7]
    r = io.imread(path + im_name + "B04.tif")
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    I = np.stack((r, g, b, nir), axis=2).astype("float32")

    if normalise_img:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq20(path, normalise_img):
    """Read cropped Sentinel-2 image: bands with resolution less than or equals to 20m."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    I = np.stack((r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3), axis=2).astype(
        "float32"
    )

    if normalise_img:
        I = (I - I.mean()) / I.std()

    return I


def read_sentinel_img_leq60(path, normalise_img):
    """Read cropped Sentinel-2 image: all bands."""
    im_name = os.listdir(path)[0][:-7]

    r = io.imread(path + im_name + "B04.tif")
    s = r.shape
    g = io.imread(path + im_name + "B03.tif")
    b = io.imread(path + im_name + "B02.tif")
    nir = io.imread(path + im_name + "B08.tif")

    ir1 = adjust_shape(zoom(io.imread(path + im_name + "B05.tif"), 2), s)
    ir2 = adjust_shape(zoom(io.imread(path + im_name + "B06.tif"), 2), s)
    ir3 = adjust_shape(zoom(io.imread(path + im_name + "B07.tif"), 2), s)
    nir2 = adjust_shape(zoom(io.imread(path + im_name + "B8A.tif"), 2), s)
    swir2 = adjust_shape(zoom(io.imread(path + im_name + "B11.tif"), 2), s)
    swir3 = adjust_shape(zoom(io.imread(path + im_name + "B12.tif"), 2), s)

    uv = adjust_shape(zoom(io.imread(path + im_name + "B01.tif"), 6), s)
    wv = adjust_shape(zoom(io.imread(path + im_name + "B09.tif"), 6), s)
    swirc = adjust_shape(zoom(io.imread(path + im_name + "B10.tif"), 6), s)

    I = np.stack(
        (r, g, b, nir, ir1, ir2, ir3, nir2, swir2, swir3, uv, wv, swirc), axis=2
    ).astype("float32")

    if normalise_img:
        I = (I - I.mean()) / I.std()

    return I


def adjust_shape(I, s):
    """Adjust shape of grayscale image I to s."""

    # crop if necesary
    I = I[: s[0], : s[1]]
    si = I.shape

    # pad if necessary
    p0 = max(0, s[0] - si[0])
    p1 = max(0, s[1] - si[1])

    return np.pad(I, ((0, p0), (0, p1)), "edge")


def reshape_for_torch(I):
    """Transpose image for PyTorch coordinates."""
    #     out = np.swapaxes(I,1,2)
    #     out = np.swapaxes(out,0,1)
    #     out = out[np.newaxis,:]
    out = I.transpose((2, 0, 1))
    return torch.from_numpy(out)
