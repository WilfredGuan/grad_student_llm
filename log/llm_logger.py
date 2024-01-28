import os
import pandas as pd
import time
import logging


class LLMLogger:
    def __init__(self, log_dir):
        # log dir
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "log_message.txt")
        if not os.path.exists(self.log_file):
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

    def close(self):
        self.log_file.close()

    def draw_result(self):
        pass

    def show_result(self):
        pass


if __name__ == "__main__":
    logger = LLMLogger("./log")
    logger.log_message_txt("test")
    logger.close()
