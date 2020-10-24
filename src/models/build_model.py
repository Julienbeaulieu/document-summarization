import torch
from torch import nn
from ..utils.registry import Registry

MODEL_ARCH_REGISTRY = Registry()


def build_model(model_cfg: dict) -> nn.Module:
    """
    build model
    :param model_cfg: model config
    :return: model
    """
    arch = model_cfg['model_arch']
    model = MODEL_ARCH_REGISTRY.get(arch)(model_cfg)  # type: ignore
    model[0].to(torch.device(model_cfg['device']))

    return model
