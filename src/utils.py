from rouge_score import rouge_scorer
from typing import Dict, List


def calculate_rouge_scores(targets: List, preds: List) -> Dict:

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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.


def _register_generic(module_dict, module_name, module):
    assert module_name not in module_dict
    module_dict[module_name] = module


class Registry(dict):
    '''
    A helper class for managing registering modules, it extends a dictionary
    and provides a register functions.
    Eg. creating a registry:
        some_registry = Registry({"default": default_module})
    There're two ways of registering new modules:
    1): normal way is just calling register function:
        def foo():
            ...
        some_registry.register("foo_module", foo)
    2): used as decorator when declaring the module:
        @some_registry.register("foo_module")
        @some_registry.register("foo_modeul_nickname")
        def foo():
            ...
    Access of module is just like using a dictionary, eg:
        f = some_registry["foo_modeul"]
    '''
    def __init__(self, *args, **kwargs):
        super(Registry, self).__init__(*args, **kwargs)

    def register(self, module_name, module=None):
        # used as function call
        if module is not None:
            _register_generic(self, module_name, module)
            return

        # used as decorator
        def register_fn(fn):
            _register_generic(self, module_name, fn)
            return fn

        return register_fn
