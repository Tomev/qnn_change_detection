from typing import Any, Dict, Optional

import mlflow
import mlflow.entities


def save_exp_conf(exp_conf: Dict[str, Any]) -> None:
    for key in exp_conf.keys():
        mlflow.log_dict(exp_conf[key], "config_" + key)


def get_run_config(run_id: str) -> Dict[str, Any]:
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri

    exp_config = {
        "general": mlflow.artifacts.load_dict(artifact_uri + "/config_general"),
        "metaheuristic": mlflow.artifacts.load_dict(
            artifact_uri + "/config_metaheuristic"
        ),
    }

    return exp_config


def get_model_state_dict(run_id: str) -> Dict[str, Any]:
    run = mlflow.get_run(run_id)
    artifact_uri = run.info.artifact_uri

    return mlflow.artifacts.load_dict(artifact_uri + "/model")


def get_exp_id(exp_conf: Dict[str, Any], exp_type: str) -> Any:
    return _ensure_created(exp_conf, exp_type).experiment_id


def _ensure_created(
    exp_conf: Dict[str, Any], exp_type: str
) -> mlflow.entities.Experiment:
    name: str = _get_exp_name(exp_conf)

    # https://mlflow.org/docs/latest/getting-started/logging-first-model/step3-create-experiment.html
    if not mlflow.get_experiment_by_name(name):
        mlflow.create_experiment(name=name, tags=_get_tags(exp_conf, exp_type))

    experiment: Optional[mlflow.entities.Experiment] = mlflow.get_experiment_by_name(
        name
    )

    if isinstance(experiment, mlflow.entities.Experiment):
        return experiment
    else:
        raise ValueError("Experiment not created!")


def _get_exp_name(exp_conf: Dict[str, Any]) -> str:
    name: str = "SIPwQNN_Manish_Rerun"

    if exp_conf["metaheuristic"]["initialization"]:
        name += "meta_init-"

    if exp_conf["metaheuristic"]["tuning"]:
        name += "meta-tuning-"

    if name == "SIPwQNN:":
        name += "baseline-"

    name = name[:-1]  # remove last -

    return name


def _get_tags(config: Dict[str, Any], exp_type: str) -> Dict[str, str]:
    tags: Dict[str, str] = {"project": "SIPwQNN", "exp_type": exp_type}

    if config["metaheuristic"]["training"]:
        tags["training"] = "metaheuristic"
    else:
        tags["training"] = "gradient"

    return tags
