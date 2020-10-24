import os
import torch

from ..envpath import AllPaths

# Get the dataset/raw path from AllPaths class
weights_path = AllPaths.trained_weights

# We can overwrite these settings using YAML files. All YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

hyperparameter_defaults = {

    'description': 'Default config from the Singleton',

    #########################
    # Dataset related configs
    #########################
    'dataset': {
        'training': 'news_training_128.p',  # news_training.p
        'validation': 'news_validation_32.p',  # news_validation.p
        'test': ''
    },

    ##########################
    # Training related configs
    ##########################
    'training': {
        'train_batch_size': 2,
        'valid_batch_size': 2,
        'train_epochs': 1,
        'learning_rate': 1e-4,
        'seed': 42,
        'max_len': 512,
        'summary_len': 150
    },

    #######################
    # Model related configs
    #######################
    'model': {
        'model_arch': 'bart',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'max_len': 120,
        'num_beams': 3,
        'repetition_penalty': 2.5,
        'length_penalty': 2.0,
        'early_stopping': True,

        # T5 Configs
        't5': {
            'pretrained': 't5-base',
            'tok_pretrained': 't5-base'
        },

        # BART Configs
        'bart': {
            'pretrained': 'facebook/bart-large-cnn',  # './src/weights/bart_large_cnn'
            'tok_pretrained': 'facebook/bart-large-cnn'  # './src/weights/bart_large_cnn'  
        }
    },

    ######################
    # Path related configs
    ######################
    'path': {
        'state_fpath': os.path.join(weights_path, 'model_01_epochs')
    }

}


def get_cfg_defaults():
    """
    Return a dict with default values
    """
    return hyperparameter_defaults.copy()

