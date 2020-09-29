from rouge_score import rouge_scorer


def calculate_rouge_scores(targets, preds):

    # TODO: Make this function more efficient

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(target, pred) for target, pred in zip(targets, preds)]

    rouge1_f1, rouge2_f1, rougeL_f1 = 0.0, 0.0, 0.0

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
