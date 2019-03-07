import os
import sys
import json

import torch
from torch import nn, optim
torch.backends.cudnn.benchmark=True

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

__all__ = [
	'save_result',
]

def save_result(result, log_dir, filename):
	path = os.path.join(log_dir, filename)
	dir = os.path.dirname(path)
	os.makedirs(dir, exist_ok=True)

	with open(path, 'w') as f:
		f.write(json.dumps(result, indent=4))