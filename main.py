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
    """
    model_name = "mistralai/Mistral-7B-Instruct-v0.2"

    # Arguments
    logger = LLMLogger(
        model_name="Mistral-7B-Instruct-v0.2", log_dir=os.getcwd() + "/log"
    )
    # print("Logger Initialized:", logger)

    model_arguemnts = ModelArguments(
        model_name_or_path=model_name,
        device=accelerator.device,
    )
    # print("Model Arguments Initialized:", model_arguemnts)

    data_arguments = DataArguments(
        # data_path=os.getcwd() + "/data/GSM8K/test.jsonl",
        # split_ratio=0.8,
        data_path=os.getcwd() + "/data/step1_1-shot/GSM8K_1-shot_final_log.jsonl",
        dataloader="Step2KnowledgeLoader",
        constructor="Step2KnowledgeConstructor",
    )
    # print("Data Arguments Initialized:", data_arguments)

    # train_arguments = TrainArguments()
    # print("Training Arguments Initialized:", train_eval_arguments)

    eval_arguments = EvalArguments(
        evaluator_name="Step2KnowledgeEvaluator",
        max_new_tokens=400,
    )

    # Load Model
    model = ModelBase(
        model_arguemnts=model_arguemnts,
        data_arguments=data_arguments,
        # train_arguments=train_arguments,
        eval_arguments=eval_arguments,
        logger=logger,
        accelerator=accelerator,
    )

    # train
    # model.train()

    # eval
    model.eval()

    # show result
    logger.close()
    logger.show_result()
