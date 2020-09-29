import wandb
import os
from pathlib import Path

weights_path = Path('/home/julien/data-science/nlp-project/weights')


def add_news_summary_configs():

    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # Defining some key variables that will be used later on in the training
    config = {
        'TRAIN_BATCH_SIZE': 2,    # input batch size for training (default: 64)
        'VALID_BATCH_SIZE': 2,    # input batch size for testing (default: 1000)
        'TRAIN_EPOCHS': 1,        # number of epochs to train (default: 10)
        'LEARNING_RATE': 1e-4,    # learning rate (default: 0.01)
        'SEED': 42,               # random seed (default: 42)
        'MAX_LEN': 512,
        'SUMMARY_LEN': 150,
        'STATE_FPATH': os.path.join(weights_path, f'model_1_epochs')
    }

    return config


    # WandB – Config is a variable that holds and saves hyperparameters and inputs
    # # Defining some key variables that will be used later on in the training
    # config = wandb.config          # Initialize config
    # config.TRAIN_BATCH_SIZE = 2    # input batch size for training (default: 64)
    # config.VALID_BATCH_SIZE = 2    # input batch size for testing (default: 1000)
    # config.TRAIN_EPOCHS = 1        # number of epochs to train (default: 10)
    # config.LEARNING_RATE = 1e-4    # learning rate (default: 0.01)
    # config.SEED = 42               # random seed (default: 42)
    # config.MAX_LEN = 512
    # config.SUMMARY_LEN = 150
    # config.STATE_FPATH = os.path.join(weights_path, f'model_{config.TRAIN_EPOCHS}_epochs')
