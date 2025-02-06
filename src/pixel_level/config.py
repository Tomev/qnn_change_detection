import os
from enum import IntEnum
from itertools import product

import numpy as np
import pennylane as qml
import torchvision.transforms as tr
from iqm.qiskit_iqm import IQMFakeApollo, IQMProvider
from niapy.algorithms.basic import CatSwarmOptimization, ParticleSwarmAlgorithm

from src.oscd_transforms import RandomFlip, RandomRot


class BandType(IntEnum):
    RGB = 0
    RGBIR = 1
    BELOW20M = 2  # 2-All bands s.t. resulution <= 20m
    ALLBANDS = 3


class QuantumDeviceType(IntEnum):
    DEFAULT = 0
    IQM_APOLLO_SIM = 1
    IQM_GARNET = 2
    IQM_CHECKER = 3
    IBM_AER = 4


class ExperimentConfigFlag(IntEnum):
    METAHEURISTIC_INITIALIZATION = 0
    METAHEURISTIC_TRAINING = 1
    METAHEURISTIC_TUNING = 2


def prepare_experiments_configs():

    n_seeds: int = 1

    for seed in range(n_seeds):
        for metaheuristic_training_config in _get_exp_metaheuristic_config():
            config = {}
            config["general"] = _get_config()
            config["general"]["SEED"] = seed
            config["metaheuristic"] = metaheuristic_training_config
            yield config


def _get_exp_metaheuristic_config():
    yield {"tuning": None, "initialization": None, "training": None}
    return  # Disable metaheuristics
    for f in list(product([True, False], repeat=3)):
        yield {
            "tuning": (
                _get_metaheuristic_tuning_config()
                if f[ExperimentConfigFlag.METAHEURISTIC_TUNING]
                else None
            ),
            "initialization": (
                _get_metaheuristic_initialization_config()
                if f[ExperimentConfigFlag.METAHEURISTIC_INITIALIZATION]
                else None
            ),
            "training": (
                _get_metaheuristic_trainig_config()
                if f[ExperimentConfigFlag.METAHEURISTIC_TRAINING]
                else None
            ),
        }


def _get_metaheuristic_initialization_config():
    conf = {
        # A Flexible Hardware/Software Codesign for Particle Swarm Optimization
        "particle_speed": 0.3 * (4 * np.pi),
        "population_size": 10,
        "generations_number": 1,
        "cutoff": 0.80,
        "algorithm": ParticleSwarmAlgorithm,
        # "algorithm": CatSwarmOptimization
    }

    return conf


def _get_metaheuristic_trainig_config():
    conf = {
        # A Flexible Hardware/Software Codesign for Particle Swarm Optimization
        "particle_speed": 0.3 * (4 * np.pi),
        "population_size": 10,
        "generations_number": 5,
        "cutoff": 0.80,
        "algorithm": ParticleSwarmAlgorithm,
        # "algorithm": CatSwarmOptimization
    }

    return conf


def _get_metaheuristic_tuning_config():
    conf = {
        "particle_speed": 0.3 * (4 * np.pi),
        "population_size": 10,
        "generations_number": 10,
        "cutoff": 0.80,
        "algorithm": ParticleSwarmAlgorithm,
        # "algorithm": CatSwarmOptimization
    }

    return conf


def _get_config():

    assert (
        _are_sipwqnn_env_vars_set()
    ), "Please set the SIPWQNN environment variables. Check config.py for more info."

    config = {
        "PATH_TO_DATASET": os.environ["SIPWQNN_DATA_PATH"],
        "DATA_TRANSFORM": tr.Compose([RandomFlip(), RandomRot()]),
        "FP_MODIFIER": 1,  # Tuning parameter, use 1 if unsure
        "BATCH_SIZE": 500,  # 32, 500 for standard Manish training
        "PATCH_SIDE": 1,
        "N_EPOCHS": 20,  # 35, 20 for standard Manish training
        "NORMALISE_IMGS": True,
        "TRAIN_STRIDE": 10,  # int(PATCH_SIDE/2) - 1
        "TYPE": BandType.BELOW20M,
        "PCA_COMPONENTS": 4,  # 4
        "N_SHOTS": 100,
        "DEVICE_TYPE": QuantumDeviceType.DEFAULT,
        "TORCH_DEVICE": "cpu",
    }

    # Or set the datapath manually, if you so desire...
    # config["PATH_TO_DATASET"] = "/path/to/dataset"

    config["NUM_BANDS"] = config["TYPE"] + 3

    n_qubits = 1 * config["PATCH_SIDE"] * config["PATCH_SIDE"] * config["NUM_BANDS"]

    if config["NUM_BANDS"] > 4:
        n_qubits = (
            1 * config["PATCH_SIDE"] * config["PATCH_SIDE"] * config["PCA_COMPONENTS"]
        )

    config["NUM_QUBITS"] = n_qubits

    _fill_dev(config)

    return config


def _fill_dev(config):
    device_getters = [
        _get_default_device,
        _get_iqm_apollo_sim_device,
        _get_iqm_garnet_device,
        _get_iqm_checker_device,
        _get_aer_sim_device,
    ]

    config["DEV"] = device_getters[config["DEVICE_TYPE"]](config)


def _get_default_device(config):
    return qml.device("default.qubit")
    # TODO TR: There's some kind of problem with the lightning.qubit device.
    # return qml.device("lightning.qubit", wires=int(2*config["NUM_QUBITS"]))


def _get_iqm_apollo_sim_device(config):
    backend = IQMFakeApollo()
    return qml.device(
        "qiskit.aer",
        wires=int(2 * config["NUM_QUBITS"]),
        backend=backend,
    )


def _get_aer_sim_device(config):
    return qml.device("qiskit.aer", wires=int(2 * config["NUM_QUBITS"]))


def _get_iqm_garnet_device(config):
    pass


def _get_iqm_checker_device(config):
    """
    Get IQM Quantum Algorithm Checker device.

    :note:
        The maximal number of shots is 1024. See
        https://iqm-finland.github.io/qiskit-on-iqm/user_guide.html.
    :note:
        The maximal number of circuits is 200. See
        https://iqm-finland.github.io/qiskit-on-iqm/user_guide.html
    :note:
        You cannot set IQM_AUTH_USERNAME and IQM_AUTH_PASSWORD in the environment
        variables to get the provider for mock garnet!

    """
    _adjust_config_for_iqm_checker(config)
    # Following the qiskit-on-iqm tutorial.
    # https://iqm-finland.github.io/qiskit-on-iqm/user_guide.html
    # Server URL taken from https://resonance.meetiqm.com/docs.
    iqm_server_url = "https://cocos.resonance.meetiqm.com/garnet:mock"
    # Remember to set the environment variables!
    # https://iqm-finland.github.io/qiskit-on-iqm/api/iqm.qiskit_iqm.iqm_provider.IQMProvider.html#iqm.qiskit_iqm.iqm_provider.IQMProvider
    assert _are_iqm_env_vars_set(), "Please set the IQM environment variables."
    provider = IQMProvider(url=iqm_server_url)
    backend = provider.get_backend()

    # Following the PennyLane_Qiskit tutorial.
    # https://docs.pennylane.ai/projects/qiskit/en/latest/
    return qml.device(
        "qiskit.aer",
        wires=int(2 * config["NUM_QUBITS"]),
        backend=backend,
        shots=config["N_SHOTS"],  # https://discuss.pennylane.ai/t/number-of-shots/562
    )


def _adjust_config_for_iqm_checker(config):
    min_shots = 1
    max_shots = 1024
    config["N_SHOTS"] = (
        config["N_SHOTS"] if min_shots <= config["N_SHOTS"] <= max_shots else max_shots
    )

    min_batch_size = 1
    max_batch_size = 200
    config["BATCH_SIZE"] = (
        config["BATCH_SIZE"]
        if min_batch_size <= config["BATCH_SIZE"] <= max_batch_size
        else max_batch_size
    )


def _are_sipwqnn_env_vars_set():
    return "SIPWQNN_DATA_PATH" in os.environ


def _are_iqm_env_vars_set():
    return "IQM_TOKEN" in os.environ
