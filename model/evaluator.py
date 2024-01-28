from typing import Any


class ModelEvaluatorBase:
    def __init__(self, model, dataloader, eval_args, logger, accelerator):
        self.model = model
        self.dataloader = dataloader
        self.evaluator_args = eval_args
        self.logger = logger
        self.accelerator = accelerator

        if self.__class__.__name__ == "ModelEvaluatorBase":
            print("You are using ModelEvaluatorBase, which is a dummy evaluator.")

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

    def _using_prompt(self, prompt):
        pass

    def _split(self):
        pass

    def _eval(self):
        pass

    def _cleanning_output(self, output):
        pass

    def _check_correctness(self, output, sample):
        pass

    def _calculate_final_score(self):
        pass


class HumanEval(ModelEvaluatorBase):
    def __init__(self, model, dataloader, eval_args, logger, accelerator):
        super().__init__(model, dataloader, eval_args, logger, accelerator)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        print("On Call")

    def _using_prompt(self, prompt):
        pass

    def _eval(self):
        pass

    def _cleanning_output(self, output):
        pass

    def _check_correctness(self, output, sample):
        pass

    def _calculate_final_score(self):
        pass


if __name__ == "__main__":
    test = HumanEval(None, None, None, None)
