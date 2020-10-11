import transformers
from yacs.config import CfgNode
from torch import nn
from typing import Tuple
from transformers import BartTokenizer, BartForConditionalGeneration

from .build_model import MODEL_ARCH_REGISTRY


@MODEL_ARCH_REGISTRY.register('bart')
def build_bart_model(model_cfg: CfgNode) -> Tuple[nn.Module, transformers.tokenization_t5.T5Tokenizer]:
    """
    Builds the t5 model using the CfgNode object,
    :param model_cfg: YAML based YACS config node
    :return: return torch neural network module
    """
    return (BartForConditionalGeneration.from_pretrained(model_cfg.BART.PRETRAINED),
            BartTokenizer.from_pretrained(model_cfg.BART.PRETRAINED))