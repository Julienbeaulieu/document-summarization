
import torch
import wandb
from time import time
from torch.utils.data import DataLoader
from typing import List, Tuple, Any

from .utils import calculate_rouge_scores


def train_model(epoch: int, tokenizer, model, device, loader: DataLoader, optimizer):
    model.train()
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids,
                        lm_labels=lm_labels)
        loss = outputs[0]

        if _ % 10 == 0:
            wandb.log({"Training Loss": loss.item()})

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate_model(tokenizer, model, device, loader: DataLoader, wandb_log=True) -> Tuple[List[Any], List[Any]]:
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            ids = data['source_ids'].to(device, dtype=torch.long)
            mask = data['source_mask'].to(device, dtype=torch.long)

            generated_ids = model.generate(input_ids=ids,
                                           attention_mask=mask,
                                           max_length=150,
                                           num_beams=2,
                                           repetition_penalty=2.5,
                                           length_penalty=1.0,
                                           early_stopping=True
                                           )
            preds = [tokenizer.decode(g, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True) for t in y]
            predictions.extend(preds)
            actuals.extend(target)

            if _ % 100 == 0:
                print(f'Completed {_}')

                # Log rouge scores
                t = time()
                eval_dict = calculate_rouge_scores(actuals, predictions)
                if wandb_log:
                    wandb.log(eval_dict)
                time_taken = time() - t
                print(f'Rouge scores: {eval_dict} \n Time taken: {time_taken}')

    return predictions, actuals, eval_dict
