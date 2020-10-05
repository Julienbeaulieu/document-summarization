
import torch
import wandb
from time import time
from torch.utils.data import DataLoader
from typing import List, Tuple, Any

from .utils import calculate_rouge_scores


def train_model(epoch: int, tokenizer, model, device, loader: DataLoader, optimizer):
    model.train()
    train_preds = []
    train_targets = []
    for _, data in enumerate(loader, 0):
        y = data['target_ids'].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone().detach()
        lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data['source_ids'].to(device, dtype=torch.long)
        mask = data['source_mask'].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids,
                        lm_labels=lm_labels)
        train_loss = outputs[0]

        preds = evaluate(model, tokenizer, ids, mask)
        target = [tokenizer.decode(t, skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True) for t in y]

        train_preds.extend(preds)
        train_targets.extend(target)

        if _ % 10 == 0:
            wandb.log({"Training Loss": train_loss.item()})
            train_eval_dict = calculate_rouge_scores(train_targets, train_preds)
            wandb.log(train_eval_dict)
            print(f'Train Rouge scores: {train_eval_dict}')

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss.item()}")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


def validate_model(tokenizer, model, device, loader: DataLoader, wandb_log=True) -> Tuple[List[Any], List[Any]]:
    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            y = data['target_ids'].to(device, dtype=torch.long)
            y_ids = y[:, :-1].contiguous()
            ids = data['source_ids'].to(device, dtype=torch.long)
            lm_labels = y[:, 1:].clone().detach()
            lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
            mask = data['source_mask'].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids,
                            lm_labels=lm_labels)
            valid_loss = outputs[0]

            preds = evaluate(model, tokenizer, ids, mask)

            # generated_ids = model.generate(input_ids=ids,
            #                                attention_mask=mask,
            #                                max_length=150,
            #                                num_beams=2,
            #                                repetition_penalty=2.5,
            #                                length_penalty=1.0,
            #                                early_stopping=True
            #                                )
            # preds = [tokenizer.decode(g, skip_special_tokens=True,
            #                           clean_up_tokenization_spaces=True) for g in generated_ids]

            target = [tokenizer.decode(t, skip_special_tokens=True,
                                       clean_up_tokenization_spaces=True) for t in y]
            val_preds.extend(preds)
            val_targets.extend(target)

            if _ % 10 == 0:
                print(f'Completed {_}')
                wandb.log({"Valid Loss": valid_loss.item()})
                # Log rouge scores
                t = time()
                eval_dict = calculate_rouge_scores(val_targets, val_preds)
                if wandb_log:
                    wandb.log(eval_dict)
                time_taken = time() - t
                print(f'Valid Rouge scores: {eval_dict} \n Time taken: {time_taken}')

    return val_preds, val_targets, eval_dict


def evaluate(model, tokenizer, ids, mask=None):
    generated_ids = model.generate(input_ids=ids,
                                   attention_mask=mask,
                                   max_length=150,
                                   num_beams=2,
                                   repetition_penalty=2.5,
                                   length_penalty=1.0,
                                   early_stopping=True
                                   )

    return [tokenizer.decode(g, skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in generated_ids]
