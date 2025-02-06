# qnn_change_detection

## Project Goals
To implement semantic segmentation using QNN and PyTorch.

## The Dataset

The Onera Satellite Change Detection dataset addresses the issue of detecting changes between satellite images from different dates.

It comprises 24 pairs of multispectral images taken from the Sentinel-2 satellites between 2015 and 2018. Locations are picked all over the world, in Brazil, USA, Europe, Middle-East and Asia. For each location, registered pairs of 13-band multispectral satellite images obtained by the Sentinel-2 satellites are provided. Images vary in spatial resolution between 10m, 20m and 60m.

Pixel-level change ground truth is provided for 14 of the image pairs. The annotated changes focus on urban changes, such as new buildings or new roads. These data can be used for training and setting parameters of change detection algorithms.

Source: https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection

## Classification Approaches
- [ ] Pixel level Classification - Files:  trai.py, oscd_dataloader.py, quantum_classifier.py
- [ ] Patch level Classification - Files:  train_patch.py, oscd_patch_dataloader.py, qcnn_classifier.py

## Getting started

The project can run pixel-level and patch segmentation. To do that, one 
has to first follow the steps below. 
- Download the repository.
- Download required packages. (TODO: prepare requirements)
- Download the dataset (see above).
- Prepare config files (see below). 
- Run the code (see below).

### Config files

To prepare the config file for an experiment, one has to prepare the
`config.py` file. The easiest way to do this is to use the prepared
`config.py.example` file. In the selected experiment folder, copy the 
`config.py.example` file content to the `config.py` file
```
cp config.py.example config.py
```
and then update the `PATH_TO_DATASET` field with the proper path to the
dataset. You can also adjust the experiment configuration according to 
your needs. After that you can run the experiment.

### Running the code

Running the experiment is fairly simple. If the configuration was done
correctly, you just need to run
```
python -m src.<selected_experiment>.train
```
and wait. By the end, the experiment will yield the training data, the
segmentation of the pictures in the dataset and the model parameters
(enabling one to reconstruct and reuse the model). 

## Remarks

Here are some issues we know, but didn't figure out.

- While `pennylane` and `qiskit` implementations of our model 
are meant to do the same computations, they do not. Their 
outputs are different. We were not able to assert why.

- On Windows machines, we noticed an error related to OMP. 
Following the instructions, hinted by the error itself,
fix the problem.
```
OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

OMP: Hint This means that multiple copies of the OpenMP runtime have been linked into the program. That is dangerous, since it can degrade performance or cause incorrect results. The best thing to do is to ensure that only a single OpenMP runtime is linked into the process, e.g. by avoiding static linking of the OpenMP runtime in any library. As an unsafe, unsupported, undocumented workaround you can set the environment variable KMP_DUPLICATE_LIB_OK=TRUE to allow the program to continue to execute, but that may cause crashes or silently produce incorrect results. For more information, please see http://www.intel.com/software/products/support/.
```
