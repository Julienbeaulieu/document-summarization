import itertools
import pandas as pd
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, BertWordPieceTokenizer

from .envpath import AllPaths

raw_data_path = AllPaths.raw
weights_path = AllPaths.trained_weights


def train_tokenizer():
    news_df = pd.read_csv(raw_data_path / 'news_summary.csv', encoding='latin-1')
    news_more_df = pd.read_csv(raw_data_path / 'news_summary_more.csv', encoding='latin-1')

    news_text = [str(t) for t in news_df.ctext]
    news_more_text = [str(t) for t in news_more_df.text]
    combined_text = [news_text, news_more_text]

    final_text = list(itertools.chain(*combined_text))
    print(type(final_text))
    tokenizer = ByteLevelBPETokenizer()
    tokenizer = CharBPETokenizer()

    tokenizer.train(['oh hi my name is Julien'], vocab_size=30_000, min_frequency=2, special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ])

    tokenizer.save_model(weights_path, "news_tokenizer")


if __name__ == '__main__':
    train_tokenizer()
