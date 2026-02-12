import argparse
import collections
import copy
import json
import logging
import math
import os
import random
import re
import signal
import ssl
import time
import urllib.request
from datetime import datetime
from os import path as osp
from tqdm import tqdm
#from solver import solve_isotropic_covariance

import numpy as np

# Blind torch
try:
    import torch
    import torchvision
    import torch.distributions as distributions
except ImportError:
    torch = None
    torchvision = None
    distributions = None

logger = logging.getLogger(__name__)



class AvgMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.
        self.n = 0
        self.avg = 0.

    def update(self, val, n):
        assert n > 0
        self.val += val * n
        self.n += n
        self.avg = self.val / self.n

    def get(self):
        return self.avg
    


def evaluate(bottom_model_A, bottom_model_B, top_model, data_loader, args):
    bottom_model_A.eval()
    bottom_model_B.eval()
    top_model.eval()
    test_acc = AvgMeter()
    test_acc.reset()
    with torch.no_grad():
        for step, (batch_x1, batch_x2, batch_y) in tqdm(enumerate(data_loader)):
            batch_x1, batch_x2, batch_y = batch_x1.to(args.device), batch_x2.to(args.device), batch_y.to(args.device)
        
            batch_x1 = batch_x1.float()
            batch_x2 = batch_x2.float()
            output_tensor_bottom_model_a = bottom_model_A(batch_x1)
            output_tensor_bottom_model_b = bottom_model_B(batch_x2)
            output = top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            test_acc.update(acc.item(), batch_y.size(0))
    
        return test_acc.get()
    