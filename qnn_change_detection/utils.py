import torch


def get_model_params_info(model: torch.nn.Module) -> None:
    print("Params stats:")
    parameters = list(model.parameters())
    print(parameters)
    print([torch.min(p) for p in parameters])
    print([torch.max(p) for p in parameters])
