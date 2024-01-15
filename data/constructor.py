import json
import re


class Constructor:
    def __init__(self, data_path, *kwargs):
        self.data_path = data_path

    def get_data(self):
        pass

    def _get_synthetic_data(self):
        pass


class GSM8KConstructor(Constructor):
    def __init__(self, data_path):
        super().__init__(data_path)

        self.data_path = data_path

    pass
