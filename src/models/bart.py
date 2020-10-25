import transformers
from torch import nn
from typing import Tuple
from transformers import BartTokenizer, BartForConditionalGeneration

from .build_model import MODEL_ARCH_REGISTRY


@MODEL_ARCH_REGISTRY.register("bart")
def build_bart_model(
    model_cfg: dict,
) -> Tuple[nn.Module, transformers.tokenization_t5.T5Tokenizer]:
    """
    Builds the t5 model using the CfgNode object,
    :param model_cfg: YAML based config
    :return: return torch neural network module
    """
    return (
        BartForConditionalGeneration.from_pretrained(model_cfg["bart"]["pretrained"]),
        BartTokenizer.from_pretrained(model_cfg["bart"]["pretrained"]),
    )
