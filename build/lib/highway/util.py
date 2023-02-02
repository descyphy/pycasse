import json
import numpy as np
import os
import sys

def wrap_to_pi(x: float, positive: bool = False) -> float:
    if positive:
        return (x) % (2 * np.pi)
    else:
        return ((x + np.pi) % (2 * np.pi)) - np.pi

def write_json(data, file_name):
    with open(file_name, 'w') as f:
        json.dump(data, f)

def read_json(file_name):
    with open(file_name, 'r') as f:
        res = json.load(f)
    return res

# Class for suppressing print-outs
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
