from yacs.config import CfgNode
from torch import nn
from ..utils import Registry

MODEL_ARCH_REGISTRY = Registry()


def build_model(model_cfg: CfgNode) -> nn.Module:
    """
    build model
    :param model_cfg: model config
    :return: model
    """
    arch = model_cfg.MODEL_ARCH
    model = MODEL_ARCH_REGISTRY.get(arch)(model_cfg)  # type: ignore

    return model
