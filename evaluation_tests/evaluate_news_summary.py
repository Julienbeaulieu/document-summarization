from src.engine import evaluate
from src.data.news_dataset import build_news_loader
from src.configs.default_configs import get_cfg_defaults
from src.models.build_model import build_model
from src.envpath import AllPaths

import pickle
import unittest
from time import time
import sys

sys.path.append('../')


processed_data_path = AllPaths.processed


class TestEvaluateNewsSummary(unittest.TestCase):
    def test_evaluate(self):
        cfg = get_cfg_defaults()

        model, tokenizer = build_model(cfg['model'])  # type: ignore

        valid_data = pickle.load(open(processed_data_path / cfg['dataset']['validation'], 'rb'))
        val_loader = build_news_loader(valid_data, tokenizer, cfg['training'], False)  # type: ignore

        t = time()
        predictions, actuals, eval_dict = evaluate(cfg['model'],
                                                   tokenizer,
                                                   model,
                                                   cfg['model']['device'],
                                                   val_loader,
                                                   wandb_log=False
                                                   )  # type: ignore
        time_taken = time() - t

        print(f"""rouge1: {eval_dict['rouge1']},rouge2: {eval_dict['rouge2']},
              rougeL: {eval_dict['rougeL']}, time_taken: {time_taken}""")

        self.assertGreater(eval_dict['rouge1'], 0.25)
        self.assertGreater(eval_dict['rouge2'], 0.15)
        self.assertGreater(eval_dict['rougeL'], 0.25)
        self.assertGreater(time_taken, 10)
