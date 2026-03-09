import json
import yaml
import torch
import numpy as np
import random

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)    
    
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)


def load_yaml(yaml_file):
    ''' Loading the configurations from a yaml file'''  

    with open(yaml_file, 'r') as file:
        args = yaml.safe_load(file)          
    
    return dict2obj(args)

# Set seeds
def set_seeds(seed: int=2025):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2025.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
