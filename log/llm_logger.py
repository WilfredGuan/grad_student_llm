import os
import pandas as pd
import time
import logging
import json


class LLMLogger:
    def __init__(self, model_name, log_dir):
        self.model_name = model_name
        # log dir
        self.log_dir = log_dir + "/" + self.model_name
        os.makedirs(os.path.dirname(self.log_dir), exist_ok=True)
        self.log_file = os.path.join(self.log_dir, "log_message.txt")
        if not os.path.exists(self.log_file):
            os.mkdir(self.log_dir)
            self.log_file = open(self.log_file, "w")
        else:
            self.log_file = open(self.log_file, "a")

        # time format
        self.log_date = time.strftime("%H:%M:%S")
        self.exp_start = time.time()
        self.log_file.write(
            f"-" * 20 + "\n" + f"Train and log at {self.log_date}" + "\n"
        )
        self.log_file.flush()

    def log_process_table(self, epoch, train_loss, train_acc, test_loss, test_acc):
        pass

    def log_message_txt(self, message):
        self.log_file.write(message + "\n")
        self.log_file.flush()

    def log_in_jsonl(self, name, message, path=""):
        if path == "":
            path = self.log_dir
        log_file = os.path.join(path, name + "_log.jsonl")
        file = open(log_file, "a")
        file.write(json.dumps(message) + "\n")
        file.close()
        # print("Log saved at:", log_file)

    def close(self):
        self.log_file.close()

    def draw_result(self):
        pass

    def show_result(self):
        pass

    def get_log_dir(self):
        return self.log_dir


if __name__ == "__main__":
    logger = LLMLogger("./log")
    logger.log_message_txt("test")
    logger.close()
