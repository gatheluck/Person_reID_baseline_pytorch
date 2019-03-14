import os
import sys
import argparse

base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
sys.path.append(base)

#from train import train
from train_net import train
from test import test
from evaluate_rerank import evaluate_rerank
from evaluate_gpu import evaluate_gpu

from options import Options

if __name__ == '__main__':
	opt = Options().parse()
	
	train(opt)
	test(opt)
	if opt.multi:
		evaluate_gpu(opt.log_dir, opt.result_suffix)
	else:
		evaluate_rerank(opt.log_dir, opt.result_suffix)

	


