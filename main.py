from model.model import *
from model.dataclass import *
from log.llm_logger import LLMLogger

from argparse import ArgumentParser
from accelerate import Accelerator, DistributedDataParallelKwargs

import os

if __name__ == "__main__":
    # print("Cuda availability:", torch.cuda.is_available())
    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    # print("Using accelerator:", kwargs_handlers)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=kwargs_handlers)

    """
    parser = ArgumentParser()
    parser.add_argument("--logdir", type=str, default="")

    Get dataloader name
    Get dataconstructor name
    Set correct mode, e.g., step_1, step_2.1, step_2.2
    """

    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    model_arguemnts = ModelArguments(
        model_name_or_path=model_name,
        device=accelerator.device,
    )
    # print("Model Arguments Initialized:", model_arguemnts)

    data_arguments = DataArguments(
        # data_path=os.getcwd() + "/data/GSM8K/test.jsonl",
        # split_ratio=0.8,
        task_mode="step_1",
        train_path=os.getcwd() + "/data/GSM8K/train.jsonl",
        test_path=os.getcwd() + "/data/GSM8K/test.jsonl",
        dataloader="GSM8KLoader",
        constructor="GSM8KConstructor",
        construction_mode="n-shot",
        n_shot=1,
    )
    # print("Data Arguments Initialized:", data_arguments)

    eval_arguments = EvalArguments(
        evaluator_name="GSM8KEval", max_new_tokens=400, temperature=1, n_samples=10
    )

    logger = LLMLogger(
        model_name=model_name.split("/")[1], log_dir=os.getcwd() + "/log"
    )
    # print("Logger Initialized:", logger)

    # Load Model
    model = ModelBase(
        model_arguemnts=model_arguemnts,
        data_arguments=data_arguments,
        eval_arguments=eval_arguments,
        logger=logger,
        accelerator=accelerator,
    )

    # eval
    model.eval()

    # show result
    logger.close()
    logger.show_result()
