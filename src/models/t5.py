import transformers
from yacs.config import CfgNode
from torch import nn
from typing import Tuple
from transformers import T5Tokenizer, T5ForConditionalGeneration

from .build_model import MODEL_ARCH_REGISTRY


@MODEL_ARCH_REGISTRY.register('t5')
def build_t5_model(model_cfg: CfgNode) -> Tuple[nn.Module, transformers.tokenization_t5.T5Tokenizer]:
    """
    Builds the t5 model using the CfgNode object,
    :param model_cfg: YAML based YACS config node
    :return: return torch neural network module
    """
    return (T5ForConditionalGeneration.from_pretrained(model_cfg.PRETRAINED),
            T5Tokenizer.from_pretrained(model_cfg.TOK_PRETRAINED))
