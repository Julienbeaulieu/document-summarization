import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from yacs.config import CfgNode
from pathlib import Path

data_path = Path('/home/nasty/document-summarization/dataset/processed')


class NewsDataset(Dataset):

    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.text = self.data.text
        self.ctext = self.data.ctext

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        ctext = str(self.ctext[index])
        ctext = ' '.join(ctext.split())

        text = str(self.text[index])
        text = ' '.join(text.split())

        source = self.tokenizer.batch_encode_plus([ctext],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt',
                                                  truncation=True
                                                  )
        target = self.tokenizer.batch_encode_plus([text],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  return_tensors='pt',
                                                  truncation=True
                                                  )

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long),
            'source_mask': source_mask.to(dtype=torch.long),
            'target_ids': target_ids.to(dtype=torch.long),
            'target_ids_y': target_mask.to(dtype=torch.long)
        }


def build_news_loader(df: pd.DataFrame, tokenizer, cfg: CfgNode, is_training: bool) -> DataLoader:
    """
    generate data loader
    :param df: DataFrame
    :tokenizer: model tokenizer
    :param config: weights and biases config object
    :param is_training: whether training
    :return: data loader
    """

    if is_training:
        batch_size = cfg.TRAIN_BATCH_SIZE
    else:
        batch_size = cfg.VALID_BATCH_SIZE

    dataset = NewsDataset(df, tokenizer, cfg.MAX_LEN, cfg.SUMMARY_LEN)

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=is_training)
    return data_loader
