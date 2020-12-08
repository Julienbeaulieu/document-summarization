import os
import torch
import ntpath

from ..envpath import AllPaths

# Get the dataset/raw path from AllPaths class
weights_path = AllPaths.trained_weights
experiment_path = AllPaths.experiments

# We can overwrite these settings using YAML files. All YAML variables MUST BE defined here first
# as this is the master list of ALL attributes.

hyperparameter_defaults = {
    "description": "Default config from the Singleton",
    
    #########################
    # Dataset related configs
    #########################
    "dataset": {
        "training": "news_training_128.p",  # news_training.p
        "validation": "news_validation_32.p",  # news_validation.p
        "test": "",
    },
    ##########################
    # Training related configs
    ##########################
    "training": {
        "train_batch_size": 2,
        "valid_batch_size": 2,
        "train_epochs": 1,
        "learning_rate": 1e-4,
        "seed": 42,
        "max_len": 512,
        "summary_len": 150,
    },
    #######################
    # Model related configs
    #######################
    "model": {
        "model_arch": "bart",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "max_len": 120,
        "num_beams": 3,
        "repetition_penalty": 2.5,
        "length_penalty": 2.0,
        "early_stopping": True,
        # T5 Configs
        "t5": {"pretrained": "t5-base", "tok_pretrained": "t5-base"},
        # BART Configs
        "bart": {
            "pretrained": "facebook/bart-large-cnn",  # './src/weights/bart_large_cnn'
            "tok_pretrained": "facebook/bart-large-cnn",  # './src/weights/bart_large_cnn'
        },
        "scheduler": "constant_schedule_with_warmup",
        "scheduler_warmup_steps": 800
    },
    ######################################
    # Path and  experiment related configs
    ######################################
    "path": {
        "state_fpath": os.path.join(weights_path, "model_base"),
        "experiment_name": "",
    },
}


def get_cfg_defaults(path=None):
    """
    Create a directory to save the weights of the model
    Return a dict with default values
    """
    if path:
        if not os.path.exists(path):
            os.mkdir(path)

        filename = path_leaf(path)
        hyperparameter_defaults["path"]["state_fpath"] = os.path.join(
            weights_path, filename
        )
        hyperparameter_defaults["path"]["experiment_name"] = filename

    return hyperparameter_defaults.copy()


def path_leaf(path):
    """
    Extract file name from path, also remove extension file name
    To be used as the name of the directory for the saved weights
    Compatible with Linux and Windows
    """
    head, tail = ntpath.split(path)
    tail = os.path.splitext(tail)[0]
    return tail or os.path.splitext(ntpath.basename(head))[0]
