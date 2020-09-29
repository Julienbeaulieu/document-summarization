import pickle
import click
import pandas as pd
from pathlib import Path

data_path = Path('/home/julien/data-science/nlp-project/dataset')


@click.command()
@click.option('--size', default=0, help="create a sample training and validation set")
def create_datasets(size: int = 0, train_test_split: float = 0.8):

    df = pd.read_csv(data_path / 'raw/news_summary.csv', encoding='latin-1')
    df = df[['text', 'ctext']]
    df.ctext = 'summarize: ' + df.ctext

    # If n_sample_size not specified, use entire data set.
    # If too large, use entire data set.
    if size == 0 or int(size) > df.shape[0]:
        size = len(df)

    n_size = round(size * train_test_split)
    df = df[:n_size]

    train_dataset = df.sample(frac=train_test_split, random_state=42)
    val_dataset = df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALIDATION Dataset: {}".format(val_dataset.shape))

    with open(data_path / f'processed/news_training_{train_dataset.shape[0]}.p', "wb") as f:
        pickle.dump(train_dataset, f)
    with open(data_path / f'processed/news_validation_{val_dataset.shape[0]}.p', "wb") as f:
        pickle.dump(val_dataset, f)


if __name__ == '__main__':
    create_datasets()
