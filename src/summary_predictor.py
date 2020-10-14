
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
                                     max_length=cfg.MAX_LENGTH,
                                     num_beams=cfg.NUM_BEAMS,
                                     repetition_penalty=cfg.REPETITION_PENALTY,
                                     length_penalty=cfg.LENGTH_PENALTY,
                                     early_stopping=cfg.EARLY_STOPPING)

        return [self.tokenizer.decode(g,
                                      skip_special_tokens=True,
                                      clean_up_tokenization_spaces=True) for g in output][0]

    def generate_long_summary(self, nested_sentences, device):
        '''Generate summary on text with <= 1024 tokens'''

        summaries = []
        for nested in nested_sentences:
            input_tokenized = self.tokenizer.encode(' '.join(nested), truncation=True, return_tensors='pt')
            input_tokenized = input_tokenized.to(device)
            summary_ids = self.model.to(device).generate(input_tokenized,
                                                         length_penalty=3.0,
                                                         min_length=30,
                                                         max_length=100)
            output = [self.tokenizer.decode(g,
                                            skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False) for g in summary_ids]
            summaries.append(output)
        summaries = [sentence for sublist in summaries for sentence in sublist]
        print(summaries)
        return summaries
