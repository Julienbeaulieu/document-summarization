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
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration

from .data.news_dataset import NewsDataset
from .engine import train_model, validate_model
from .configs.news_summary_configs import add_news_summary_configs

# TODO: Create env file and use Environs library to handle local vars
data_path = Path('/home/julien/data-science/nlp-project/dataset/processed')

device = 'cuda' if cuda.is_available() else 'cpu'


def main():
    # WandB – Initialize a new run
    configs = add_news_summary_configs()    
    
    wandb.init(project="transformers_tutorials_summarization", config=configs)

    config = wandb.config

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(config.SEED)  # pytorch random seed
    np.random.seed(config.SEED)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    # tokenzier for encoding the text
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    df_train = pickle.load(open(data_path / 'news_training_128.p', 'rb'))
    df_valid = pickle.load(open(data_path / 'news_validation_32.p', 'rb'))

    training_set = NewsDataset(df_train, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)
    val_set = NewsDataset(df_valid, tokenizer, config.MAX_LEN, config.SUMMARY_LEN)

    # Defining the parameters for creation of dataloaders
    train_params = {
        'batch_size': config.TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 0
        }

    val_params = {
        'batch_size': config.VALID_BATCH_SIZE,
        'shuffle': False,
        'num_workers': 0
        }

    # Creation of Dataloaders for testing and validation. This will be used down for training and
    # validation stage for the model.
    training_loader = DataLoader(training_set, **train_params)
    val_loader = DataLoader(val_set, **val_params)

    model = T5ForConditionalGeneration.from_pretrained("t5-base")
    model = model.to(device)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=config.LEARNING_RATE)

    # Log metrics with wandb
    wandb.watch(model, log="all")
    # Training loop
    print('Initiating Fine-Tuning for the model on our dataset')

    for epoch in range(config.TRAIN_EPOCHS):
        train_model(epoch, tokenizer, model, device, training_loader, optimizer)
        predictions, actuals = validate_model(epoch, tokenizer, model, device, val_loader)
    
    # Save model weights
    print(f"Saving the model {epoch}")
    model.save_pretrained(config.STATE_FPATH)
 
    print('Now generating summaries on our fine tuned model for the validation dataset and saving it in a dataframe')

    final_df = pd.DataFrame({'Generated Text': predictions, 'Actual Text': actuals})
    final_df.to_csv('./models/predictions.csv')
    print('Output Files generated for review')
    

if __name__ == '__main__':
    main()
