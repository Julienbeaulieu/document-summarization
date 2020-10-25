"""
Run a basic experiment which fine tunes T5 or BART model from hugging face using news_summary
dataset. Run validation, get average Rouge scores on predictions and save predictions to csv
"""

import wandb
import argparse
import pandas as pd
import numpy as np
import pickle
import yaml
import torch
import transformers
from typing import Dict

from .engine import train_model, evaluate
from .data.news_dataset import build_news_loader
from .configs.default_configs import get_cfg_defaults
from .models.build_model import build_model
from .envpath import AllPaths

# Get the dataset/processed path from AllPaths class
processed_data_path = AllPaths.processed


def main(
    cfg: Dict,
    yaml_config: yaml,
    save_weights: bool = True,
    generate_csv_preds: bool = True,
):

    # WandB – Initialize a new run
    wandb.init(project="nyt_summarization_app", config=cfg)

    # Merge yaml config file with cfg
    wandb.config.update(yaml_config, allow_val_change=True)
    cfg = wandb.config

    # Set random seeds and deterministic pytorch for reproducibility
    torch.manual_seed(cfg["training"]["seed"])  # pytorch random seed
    np.random.seed(cfg["training"]["seed"])  # numpy random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_data = pickle.load(
        open(processed_data_path / cfg["dataset"]["training"], "rb")
    )
    valid_data = pickle.load(
        open(processed_data_path / cfg["dataset"]["validation"], "rb")
    )

    # Get model and tokenzier for encoding the text
    model, tokenizer = build_model(cfg["model"])  # type: ignore

    train_loader = build_news_loader(train_data, tokenizer, cfg["training"], True)  # type: ignore
    val_loader = build_news_loader(valid_data, tokenizer, cfg["training"], False)  # type: ignore

    optimizer = transformers.AdamW(params=model.parameters(), lr=cfg["training"]["learning_rate"])  # type: ignore

    scheduler = transformers.get_constant_schedule_with_warmup(optimizer, 800)

    # Log metrics with wandb
    wandb.watch(model, log="all")  # type: ignore

    # Training loop
    print("Initiating Fine-Tuning for the model on our dataset")

    for epoch in range(cfg["training"]["train_epochs"]):
        train_model(
            cfg["model"],
            epoch,
            tokenizer,
            model,
            cfg["model"]["device"],
            train_loader,
            optimizer,
            scheduler
        )  # type: ignore

        predictions, actuals, eval_dict = evaluate(
            cfg["model"],
            tokenizer,  # type: ignore
            model,  # type: ignore
            cfg["model"]["device"],
            val_loader,
        )  # type: ignore

    # Save model weights
    if save_weights:
        print(f"Saving the model at {cfg['path']['state_fpath']}")
        model.save_pretrained(cfg["path"]["state_fpath"])  # type: ignore

    if generate_csv_preds:
        print(
            "Now generating summaries on fine tuned model for the validation dataset and saving it in a dataframe"
        )

        final_df = pd.DataFrame({"Generated Text": predictions, "Actual Text": actuals})  # type: ignore
        final_df.to_csv("./outputs/predictions.csv")
        print("Output Files generated for review")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment_filename",
        nargs="?",
        type=str,
        help="Filename of the YAML experiment file",
    )
    args = parser.parse_args()

    if args.experiment_filename:
        with open(args.experiment_filename) as file:
            yaml_config = yaml.load(file, Loader=yaml.FullLoader)
    else:
        yaml_config = {}

    cfg = get_cfg_defaults(args.experiment_filename)

    main(cfg, yaml_config)
