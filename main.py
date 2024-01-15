from data.loader import *
from data.constructor import *

from model.model import *
from model.trainer import *
from model.evaluator import *

from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM
from log.llm_logger import LLMLogger

# Arguments
logger = LLMLogger("./log")
print("Logger Initialized:", logger)

model_arguemnts = ModelArguments(
    model_name_or_path="mistralai/Mistral-7B-Instruct-v0.2"
)
print("Model Arguments Initialized:", model_arguemnts)

data_arguments = DataArguments(data_path="models/grad_std_llm/data/GSM8K/test.jsonl")
print("Data Arguments Initialized:", data_arguments)

training_arguments = TrainArguments(output_dir="models/grad_std_llm/test_dir/")
print("Training Arguments Initialized:", training_arguments)


# data
# = GSM8KLoader(data_path, batch_size=1, shuffle=True)

# Load Model
model = ModelBase(
    ModelArguments,
    DataArguments,
    TrainArguments,
)


# train

# eval

# show result
logger.close()
logger.draw_result()
