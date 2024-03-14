from ..base.BaseWrapper import BaseWrapper
from ..models.BertPunctuator import BertPunctuator
from ..preprocessors.BertPreprocessor import BertPreprocessor
from ..trainers.BertPunctuatorTrainer import BertPunctuatorTrainer


class BertPunctuatorWrapper(BaseWrapper):
    def __init__(self, config):
        super().__init__(config)

        self._config = config
        self._preprocessor = None #BertPreprocessor(config)
        self._classifier = BertPunctuator(config)
        self._trainer = BertPunctuatorTrainer(self._classifier, self._preprocessor, self._config)

    def train(self):
        self._trainer.train()

    def predict(self):
        raise NotImplementedError