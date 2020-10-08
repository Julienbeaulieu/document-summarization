
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
        output = self.model.generate(input_ids,
                                     max_length=cfg.MAX_LENGTH,
                                     num_beams=cfg.NUM_BEAMS,
                                     repetition_penalty=cfg.REPETITION_PENALTY,
                                     length_penalty=cfg.LENGTH_PENALTY,
                                     early_stopping=cfg.EARLY_STOPPING)

        return [self.tokenizer.decode(g,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=True) for g in output][0]
