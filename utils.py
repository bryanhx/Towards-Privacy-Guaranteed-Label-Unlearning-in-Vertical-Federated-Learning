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



class LoggerPrecisionFilter(logging.Filter):
    def __init__(self, precision):
        super().__init__()
        self.print_precision = precision

    def str_round(self, match_res):
        return str(round(eval(match_res.group()), self.print_precision))

    def filter(self, record):
        # use regex to find float numbers and round them to specified precision
        if not isinstance(record.msg, str):
            record.msg = str(record.msg)
        if record.msg != "":
            if re.search(r"([-+]?\d+\.\d+)", record.msg):
                record.msg = re.sub(r"([-+]?\d+\.\d+)", self.str_round,
                                    record.msg)
        return True
    



def update_logger(args, clear_before_add=True):
    import os
    import logging

    root_logger = logging.getLogger("VFU")

    # clear all existing handlers and add the default stream
    if clear_before_add:
        root_logger.handlers = []
        handler = logging.StreamHandler()
        logging_fmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(" \
                      "message)s"
        handler.setFormatter(logging.Formatter(logging_fmt))

        root_logger.addHandler(handler)

    # update level
    if args.verbose > 0:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN
        logger.warning("Skip DEBUG/INFO messages")
    root_logger.setLevel(logging_level)

    if args.outdir == "":
        args.outdir = os.path.join(os.getcwd(), "exp")
    if args.expname == "":
        args.expname = f"{args.unlearn_method}_{args.model_type}_on" \
                      f"_{args.data}_lr{args.optimizer_lr}"
    args.expname = args.expname.replace(" ", "")
    args.expname = f"{args.expname}"
    args.outdir = os.path.join(args.outdir, args.expname)
    outdir = os.path.join(args.outdir, "_sub_exp" +
                          datetime.now().strftime('_%Y%m%d%H%M%S')
                          ) 
    #outdir = os.path.join(outdir, (args.std)
    while os.path.exists(outdir):
        time.sleep(1)
        outdir = os.path.join(
            args.outdir,
            "sub_exp" + datetime.now().strftime('_%Y%m%d%H%M%S'))
    args.outdir = outdir
    # if not, make directory with given name
    os.makedirs(args.outdir, exist_ok=True)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(args.outdir, 'exp_print.log'))
    fh.setLevel(logging.DEBUG)
    logger_formatter = logging.Formatter(
        "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    fh.setFormatter(logger_formatter)
    root_logger.addHandler(fh)

    # set print precision for terse logging
    np.set_printoptions(precision=args.print_decimal_digits)
    precision_filter = LoggerPrecisionFilter(args.print_decimal_digits)
    # attach the filter to the fh handler to propagate the filter, since
    # "Filters, unlike levels and handlers, do not propagate",
    # ref https://stackoverflow.com/questions/6850798/why-doesnt-filter-
    # attached-to-the-root-logger-propagate-to-descendant-loggers
    for handler in root_logger.handlers:
        handler.addFilter(precision_filter)

    import socket
    root_logger.info(f"the current machine is at"
                     f" {socket.gethostbyname(socket.gethostname())}")
    root_logger.info(f"the current dir is {os.getcwd()}")
    root_logger.info(f"the output dir is {args.outdir}")

    root_logger.info("the used configs are: \n" + str(args))



# def save_args(args):
#     """
#         1) make the cfg attributes immutable;
#         2) save the frozen cfg_check_funcs into
#         "self.outdir/config.yaml" for better reproducibility;
#         3) if self.wandb.use=True, update the frozen config

#     :return:
#     """
#     # save the final cfg
#     with open(os.path.join(args.outdir, "config.yaml"), 'w') as outfile:
#         from contextlib import redirect_stdout
#         with redirect_stdout(outfile):
#             tmp_cfg = copy.deepcopy(args)
#             print(tmp_cfg.dump())



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
        for step, (batch_x, batch_y) in tqdm(enumerate(data_loader)):
            if args.data != 'yahoo':
                batch_x, batch_y = batch_x.to(args.device), batch_y.to(args.device)
            if args.data == 'cifar10':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif args.data == 'cifar100':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif args.data == 'mnist':
                x_a = batch_x[:, :, :, 0:14]
                x_b = batch_x[:, :, :, 14:28]
            elif args.data == 'mri':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            elif args.data == 'ct':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            elif args.data == 'yahoo':
                for i in range(len(batch_x)):
                    batch_x[i] = batch_x[i].long().cuda()
                batch_y = batch_y[0].long().cuda()
                x_b = batch_x[1]
                x_a = batch_x[0]
            else:
                raise ValueError(f'No dataset named {args.data}!')
            output_tensor_bottom_model_a = bottom_model_A(x_a)
            output_tensor_bottom_model_b = bottom_model_B(x_b)
            output = top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            test_acc.update(acc.item(), batch_y.size(0))
    
        return test_acc.get()