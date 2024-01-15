from typing import Any


class EvaluatorBase:
    def __init__(self, model, data_loader, device, args):
        self.model = model
        self.data_loader = data_loader
        self.device = device
        self.args = args

    def evaluate(self):
        raise NotImplementedError

    def get_scores(self):
        raise NotImplementedError

    def get_scores_and_labels(self):
        raise NotImplementedError

    def get_scores_and_labels_and_logits(self):
        raise NotImplementedError

    def get_scores_and_labels_and_logits_and_features(self):
        raise NotImplementedError

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError
