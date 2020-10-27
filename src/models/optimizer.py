import transformers


def build_scheduler(optimizer, cfg, num_training_steps):

    scheduler_type = cfg['scheduler']
    warmup_steps = cfg['scheduler_warmup_steps']
    if scheduler_type == 'constant_schedule_with_warmup':
        scheduler = transformers.get_constant_schedule_with_warmup(optimizer, warmup_steps)
        return scheduler

    elif scheduler_type == 'cosine_schedule_with_warmup':
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
        return scheduler

    else:
        raise Exception('Scheduler name invalid, choices are: "constant_schedule_with_warmup"' + '\n' +
                        'or "cosine_schedule_with_warmup"')
