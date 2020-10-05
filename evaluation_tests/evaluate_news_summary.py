import pickle
import unittest
from pathlib import Path
from time import time
import sys

sys.path.append('../')

from src.engine import validate_model
from src.data.news_dataset import build_news_loader
from src.configs.yacs_configs import get_cfg_defaults, add_pretrained
from src.models.build_model import build_model

# TODO: Create env file and use Environs library to handle local vars
data_path = Path('/home/nasty/document-summarization/dataset/processed')


class TestEvaluateNewsSummary(unittest.TestCase):
    def test_evaluate(self):
        cfg = get_cfg_defaults()
        add_pretrained(cfg)

        model, tokenizer = build_model(cfg.MODEL)  # type: ignore

        valid_data = pickle.load(open(data_path / 'news_validation_32.p', 'rb'))
        val_loader = build_news_loader(valid_data, tokenizer, cfg.TRAINING, False)  # type: ignore

        t = time()
        predictions, actuals, eval_dict = validate_model(tokenizer, model, cfg.MODEL.DEVICE, val_loader, wandb_log=False)  # type: ignore
        time_taken = time() - t

        print(f"rouge1: {eval_dict['rouge1']}, rouge2: {eval_dict['rouge2']}, rougeL: {eval_dict['rougeL']}, time_taken: {time_taken}")
        self.assertGreater(eval_dict['rouge1'], 0.25)
        self.assertGreater(eval_dict['rouge2'], 0.15)
        self.assertGreater(eval_dict['rougeL'], 0.25)
        # self.assertGreater(time_taken, 50)
