
from .models.build_model import build_model


class SummaryPredictor:
    """
    Summarizes a given text
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.model, self.tokenizer = build_model(self.cfg)

    def __call__(self, text, cfg):
        """
        Summarize on a single text
        """
        assert isinstance(text, str), "text must be a string"

        input_ids = self.tokenizer.encode(text, return_tensors="pt")
        input_ids = input_ids.to(cfg.DEVICE)
        output = self.model.generate(input_ids,
                                     max_length=cfg['max_len'],
                                     num_beams=cfg['num_beams'],
                                     repetition_penalty=cfg['repetition_penalty'],
                                     length_penalty=cfg['length_penalty'],
                                     early_stopping=cfg['early_stopping'])

        return [self.tokenizer.decode(g,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True) for g in output][0]

    def generate_long_summary(self, cfg, nested_sentences, device):
        '''Generate summary on text with <= 1024 tokens'''

        summaries = []
        for nested in nested_sentences:
            input_tokenized = self.tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
            input_tokenized = input_tokenized.to(device)
            summary_ids = self.model.to(device).generate(input_tokenized,
                                                         length_penalty=cfg['length_penalty'],
                                                         min_length=80,
                                                         max_length=cfg['max_len'])
            output = [self.tokenizer.decode(g,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False) for g in summary_ids]
            summaries.append(output)
        summaries = [sentence for sublist in summaries for sentence in sublist]
        return summaries
