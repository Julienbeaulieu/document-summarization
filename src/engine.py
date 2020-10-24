
import torch
import wandb
from time import time
from torch.utils.data import DataLoader
from typing import List, Tuple, Any, Dict
from rouge_score import rouge_scorer


# Fine tune model
def train_model(cfg: Dict, epoch: int, tokenizer, model, device, loader: DataLoader, optimizer):
    model.train()
    train_preds = []
    train_targets = []
    for _, data in enumerate(loader, 0):
        inputs = prepare_inputs(data, tokenizer, device)

        outputs = model(input_ids=inputs['ids'],
                        attention_mask=inputs['mask'],
                        decoder_input_ids=inputs['y_ids'],
                        labels=inputs['lm_labels'])

        train_loss = outputs[0]

        preds = predict_batch(cfg, model, tokenizer, inputs['ids'], inputs['mask'])
        targets = [tokenizer.decode(t, skip_special_tokens=True,
                                    clean_up_tokenization_spaces=True) for t in inputs['y']]

        train_preds.extend(preds)
        train_targets.extend(targets)

        if _ % 10 == 0:
            train_eval_dict = evaluate_on_rouge_scores(train_targets, train_preds)
            wandb.log({"Train Rouge Scores": train_eval_dict})
            wandb.log({"Training Loss": train_loss.item()})
            print(f'Train Rouge scores: {train_eval_dict}')

        if _ % 500 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss.item()}")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()


def evaluate(cfg: Dict,
             tokenizer,
             model,
             device,
             loader: DataLoader,
             wandb_log=True,
             eval_type="train") -> Tuple[List[Any], List[Any], Dict[Any, Any]]:
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for _, data in enumerate(loader, 0):
            inputs = prepare_inputs(data, tokenizer, device)

            outputs = model(input_ids=inputs['ids'],
                            attention_mask=inputs['mask'],
                            decoder_input_ids=inputs['y_ids'],
                            labels=inputs['lm_labels'])

            loss = outputs[0]

            batch_preds = predict_batch(cfg, model, tokenizer, inputs['ids'], inputs['mask'])

            batch_targets = [tokenizer.decode(t, skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for t in inputs['y']]
            preds.extend(batch_preds)
            targets.extend(batch_targets)

            if _ % 10 == 0:
                print(f'Completed {_}')
                # Log rouge scores
                t = time()
                eval_dict = evaluate_on_rouge_scores(targets, preds)
                if wandb_log:
                    wandb.log({"Valid Loss": loss.item()})
                    wandb.log({"Valid Rouge Scores": eval_dict})
                time_taken = time() - t
                print(f'Valid Rouge scores: {eval_dict} \n Time taken: {time_taken}')

    return preds, targets, eval_dict


def prepare_inputs(inputs, tokenizer, device):
    y = inputs['target_ids'].to(device, dtype=torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = inputs['source_ids'].to(device, dtype=torch.long)
    mask = inputs['source_mask'].to(device, dtype=torch.long)

    return {'y': y,
            'y_ids': y_ids,
            'lm_labels': lm_labels,
            'ids': ids,
            'mask': mask}


def predict_batch(cfg, model, tokenizer, ids, mask=None):
    generated_ids = model.generate(input_ids=ids,
                                   attention_mask=mask,
                                   max_length=cfg['max_len'],
                                   num_beams=cfg['num_beams'],
                                   repetition_penalty=cfg['repetition_penalty'],
                                   length_penalty=cfg['length_penalty'],
                                   early_stopping=cfg['early_stopping']
                                   )

    return [tokenizer.decode(g, skip_special_tokens=True,
                             clean_up_tokenization_spaces=True) for g in generated_ids]


def evaluate_on_rouge_scores(targets: List, preds: List) -> Dict:

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(target, pred) for target, pred in zip(targets, preds)]

    rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0

    # TODO: Very ugly - refactor needed
    for score in scores:
        for k, v in score.items():
            if k == 'rouge1':
                rouge1_f1 += v.fmeasure
            if k == 'rouge2':
                rouge2_f1 += v.fmeasure
            if k == 'rougeL':
                rougeL_f1 += v.fmeasure

    eval_dict = {
        'rouge1': rouge1_f1 / len(scores),
        'rouge2': rouge2_f1 / len(scores),
        'rougeL': rougeL_f1 / len(scores)
    }

    return eval_dict
