from typing import Any, Dict


def get_config() -> Dict[str, Any]:

    config: Dict[str, Any] = {
        "PATH_TO_DATASET": "k:/Data/OSCD/",
        "DATA_AUG": False,
        "FP_MODIFIER": 1,  # Tuning parameter, use 1 if unsure
        "BATCH_SIZE": 500,  # 32
        "PATCH_SIDE": 2,
        "N_EPOCHS": 2,  # 35
        "NORMALISE_IMGS": True,
        "TRAIN_STRIDE": 10,  # int(PATCH_SIDE/2) - 1
        # TODO TR: Use descriptions or enums instead of magic numbers.
        "TYPE": 0,  # 0-RGB | 1-RGBIr | 2-All bands s.t. resulution <= 20m | 3-All bands
        "PCA_COMPONENTS": 1,  # 4
        "LOAD_TRAINED": False,
        "SHOW_RESULTS": False,
    }

    config["NUM_BANDS"] = config["TYPE"] + 3
    config["NUM_QUBITS"] = 2 * config["PATCH_SIDE"] * config["PATCH_SIDE"]

    return config
