'''
Run a basic experiment which fine tunes T5 model from hugging face using news_summary dataset
Run validation, get average Rouge scores on predictions and save predictions to csv
'''

import wandb
import pandas as pd
import numpy as np
import pickle
import torch
from torch import cuda
from pathlib import Path
from yacs.config import CfgNode

from .engine import train_model, validate_model
from .data.news_dataset import build_news_loader
from .configs.yacs_configs import get_cfg_defaults, cfg_to_dict
from .models.build_model import build_model

# TODO: Create env file and use Environs library to handle local vars
data_path = Path('/home/nasty/document-summarization/dataset/processed')


def main(cfg: CfgNode):

    # WandB – Initialize a new run
    configs = cfg_to_dict(cfg.clone())
    wandb.init(project="transformers_tutorials_summarization", config=configs)

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(cfg.TRAINING.SEED)  # pytorch random seed
    np.random.seed(cfg.TRAINING.SEED)  # numpy random seed
    # torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text

    model, tokenizer = build_model(cfg.MODEL)  # type: ignore

    train_data = pickle.load(open(data_path / 'news_training_128.p', 'rb'))
    valid_data = pickle.load(open(data_path / 'news_validation_32.p', 'rb'))

    train_loader = build_news_loader(train_data, tokenizer, cfg.TRAINING, True)  # type: ignore
    val_loader = build_news_loader(valid_data, tokenizer, cfg.TRAINING, False)  # type: ignore

    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.TRAINING.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(cfg.TRAINING.TRAIN_EPOCHS):
        train_model(epoch, tokenizer, model, device, train_loader, optimizer)  # type: ignore
        predictions, actuals, eval_dict = validate_model(tokenizer, model, device, val_loader)  # type: ignore

    # Save model weights
    print(f"Saving the model {epoch}")
    model.save_pretrained(cfg.PATH.STATE_FPATH)

    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')

    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})  # type: ignore
    final_df.to_csv('./models/predictions.csv')
    print('Output Files generated for review')


if __name__ == '__main__':
    cfg = get_cfg_defaults()
    main(cfg)
