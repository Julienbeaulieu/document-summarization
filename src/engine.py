import torch
import wandb
from typing import Dict, List
from rouge_score import rouge_scorer


# Fine tune model
def train_model(
    cfg: Dict,
    epoch: int,
    tokenizer,
    model,
    device,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
):

    model.train()
    train_preds, train_targets = [], []
    total_train_loss = 0

    for step, batch in enumerate(train_loader, 0):
        inputs = prepare_inputs(batch, tokenizer, device)

        outputs = model(
            input_ids=inputs["ids"],
            attention_mask=inputs["mask"],
            decoder_input_ids=inputs["y_ids"],
            labels=inputs["lm_labels"],
        )

        train_loss = outputs[0]

        preds = predict_batch(cfg, model, tokenizer, inputs["ids"], inputs["mask"])

        targets = [
            tokenizer.decode(
                t, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            for t in inputs["y"]
        ]

        train_preds.extend(preds)
        train_targets.extend(targets)
        total_train_loss += train_loss

        if step % 10 == 0:
            # Evaluate on training
            train_eval_dict = evaluate_on_rouge_scores(train_targets, train_preds)

            # Log to wandb
            wandb.log({"Train Rouge Scores": train_eval_dict})
            wandb.log({"Training Loss": train_loss.item()})
            wandb.log({"lr": optimizer.param_groups[-1]["lr"]})
            print(
                f"Batch {step} of {len(train_loader)}, Train Rouge scores: {train_eval_dict}"
            )

        if step % 300 == 0:
            print(f"Epoch: {epoch}, Train loss: {train_loss.item()}")

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_train_loss / len(train_loader)
    wandb.log({"avg_train_loss": avg_train_loss})
    print(f"  Average training  loss: {avg_train_loss}")
    print("Running Validation...")

    # Beging validation
    model.eval()

    # Initialize preds and metrics to log
    valid_preds, valid_targets = [], []
    total_eval_loss = 0
    valid_rouge_dict = {
        "rouge1": 0,
        "rouge2": 0,
        "rougeL": 0,
    }

    with torch.no_grad():
        for step, batch in enumerate(val_loader, 0):
            inputs = prepare_inputs(batch, tokenizer, device)

            outputs = model(
                input_ids=inputs["ids"],
                attention_mask=inputs["mask"],
                decoder_input_ids=inputs["y_ids"],
                labels=inputs["lm_labels"],
            )

            val_loss = outputs[0]

            batch_preds = predict_batch(
                cfg, model, tokenizer, inputs["ids"], inputs["mask"]
            )

            batch_targets = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in inputs["y"]
            ]
            valid_preds.extend(batch_preds)
            valid_targets.extend(batch_targets)
            total_eval_loss += val_loss

            if step % 10 == 0:

                # Log rouge scores
                eval_dict = evaluate_on_rouge_scores(valid_targets, valid_preds)

                wandb.log({"Valid Loss": val_loss.item()})
                wandb.log({"Valid Rouge Scores": eval_dict})

                valid_rouge_dict["rouge1"] += eval_dict["rouge1"]
                valid_rouge_dict["rouge2"] += eval_dict["rouge2"]
                valid_rouge_dict["rougeL"] += eval_dict["rougeL"]

                print(f"Valid Rouge scores: {eval_dict}")

        valid_rouge_dict["rouge1"] = valid_rouge_dict["rouge1"] / (len(val_loader) / 10)
        valid_rouge_dict["rouge2"] += eval_dict["rouge2"] / (len(val_loader) / 10)
        valid_rouge_dict["rougeL"] += eval_dict["rougeL"] / (len(val_loader) / 10)

        avg_valid_loss = total_eval_loss / len(val_loader)

        wandb.log(
            {"avg_valid_loss": avg_valid_loss, "avg_rouge_scores": valid_rouge_dict}
        )

        print(f"  Average validation loss: {avg_valid_loss}")


def prepare_inputs(inputs, tokenizer, device):
    y = inputs["target_ids"].to(device, dtype=torch.long)
    y_ids = y[:, :-1].contiguous()
    lm_labels = y[:, 1:].clone().detach()
    lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    ids = inputs["source_ids"].to(device, dtype=torch.long)
    mask = inputs["source_mask"].to(device, dtype=torch.long)

    return {"y": y, "y_ids": y_ids, "lm_labels": lm_labels, "ids": ids, "mask": mask}


def predict_batch(cfg, model, tokenizer, ids, mask=None):
    generated_ids = model.generate(
        input_ids=ids,
        attention_mask=mask,
        max_length=cfg["max_len"],
        num_beams=cfg["num_beams"],
        repetition_penalty=cfg["repetition_penalty"],
        length_penalty=cfg["length_penalty"],
        early_stopping=cfg["early_stopping"],
    )

    return [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in generated_ids
    ]


def evaluate_on_rouge_scores(targets: List, preds: List) -> Dict:

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = [scorer.score(target, pred) for target, pred in zip(targets, preds)]

    rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0

    # TODO: Very ugly - refactor needed
    for score in scores:
        for k, v in score.items():
            if k == "rouge1":
                rouge1_f1 += v.fmeasure
            if k == "rouge2":
                rouge2_f1 += v.fmeasure
            if k == "rougeL":
                rougeL_f1 += v.fmeasure

    eval_dict = {
        "rouge1": rouge1_f1 / len(scores),
        "rouge2": rouge2_f1 / len(scores),
        "rougeL": rougeL_f1 / len(scores),
    }

    return eval_dict
