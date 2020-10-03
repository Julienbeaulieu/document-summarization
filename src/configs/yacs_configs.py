import os
import yacs 

from yacs.config import CfgNode as ConfigurationNode
from pathlib import Path

weights_path = Path('/home/nasty/document-summarization/weights')

# YACS overwrite these settings using YAML, all YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

__C = ConfigurationNode()

cfg = __C
__C.DESCRIPTION = 'Default config from the Singleton'
__C.TRAINING = ConfigurationNode()
__C.TRAINING.TRAIN_BATCH_SIZE = 2
__C.TRAINING.VALID_BATCH_SIZE = 2
__C.TRAINING.TRAIN_EPOCHS = 1
__C.TRAINING.LEARNING_RATE = 1e-4
__C.TRAINING.SEED = 42
__C.TRAINING.MAX_LEN = 512
__C.TRAINING.SUMMARY_LEN = 150

__C.PATH = ConfigurationNode()
__C.PATH.STATE_FPATH = os.path.join(weights_path, f'model_{__C.TRAINING.TRAIN_EPOCHS}_epochs')


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()


def cfg_to_dict(cfg):
    if isinstance(cfg, yacs.config.CfgNode):
        for k, v in cfg.items():
            if isinstance(v, yacs.config.CfgNode):
                cfg[k] = dict(v)
                cfg_to_dict(v)
    return dict(cfg)
