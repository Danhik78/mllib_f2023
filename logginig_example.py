import hashlib
import inspect
from logs.Logger import Logger
import random

def generate_experiment_name(base_functions: list, reg_coeff: float, lr: float) -> (str,str):
    # Convert base functions to string representation and hash them
    function_strings = [inspect.getsource(f).strip() for f in base_functions]
    concatenated = "\n".join(function_strings)
    hash_id = hashlib.md5(concatenated.encode()).hexdigest()[:6]  # taking the first 6 characters for brevity

    # Construct the name
    name = f"Reg{reg_coeff}_LR{lr}_FuncHash{hash_id}"

    return name, concatenated




