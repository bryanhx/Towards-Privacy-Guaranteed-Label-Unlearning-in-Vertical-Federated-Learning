# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: Tae Hong Xi
# @Institution: Universiti Malaya
# @E-mail: taehongxi55@gmail.com
# @Time: 2024/7/10
# ---

import logging
import os

import torch

import unlearn_2labels

from configs import config
from utils import update_logger, evaluate
from vfl_framework import VFL
from dataset import get_dataset
from copy import deepcopy
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms
from torch import nn
from MIA import MIA
import time
import numpy as np
import random
from model import read_data_text
logger = logging.getLogger(f"VFU.{__name__}")




def main():
    args = config()
    update_logger(args)
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True
    
    if args.model_type == 'resnet18':
        if args.data=='cifar10':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/emb_exp_resnet18_on_cifar10_lr0.001/full_train/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar10_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
            
        elif args.data == 'cifar100':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/EE_resnet18_on_cifar100_lr0.001/full_train/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_resnet18_on_cifar100_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
        else:
            raise ValueError(f'No dataset named {args.data}!')
        

    else:
        if args.data=='cifar10':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar10_lr0.001/full_train/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar10_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
            
        elif args.data == 'cifar100':
            full_bottom_model_A = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_bottom_model_A.pt", weights_only=False)
            full_bottom_model_B = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_bottom_model_B.pt", weights_only=False)
            full_top_model = torch.load("/scratch/taehongxi/exp/EE_vgg16_on_cifar100_lr0.001/full_train2/full_top_model.pt", weights_only=False)
            retrain_bottom_model_A = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
            retrain_bottom_model_B = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
            retrain_top_model = torch.load("C:/Users/Bryan/Documents/Vertical Federated Unlearning/exp/EE_vgg16_on_cifar100_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
        else:
            raise ValueError(f'No model named {args.model_type}!')
    
    
    full_bottom_model_A = full_bottom_model_A.to(args.device)
    full_bottom_model_B = full_bottom_model_B.to(args.device)
    full_top_model = full_top_model.to(args.device)
    retrain_bottom_model_A = retrain_bottom_model_A.to(args.device)
    retrain_bottom_model_B = retrain_bottom_model_B.to(args.device)
    retrain_top_model = retrain_top_model.to(args.device)

    datasets = get_dataset(args)
    trainset = deepcopy(datasets.trainset)
    testset = deepcopy(datasets.testset)

    full_train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    full_test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    class_to_forget = [0, 2]
    idx_train_forget = []
    idx_train_retain = []
    idx_test_forget = []
    idx_test_retain = []

    
    for i in range(len(trainset)):
        if trainset.targets[i] in class_to_forget:
            idx_train_forget.append(i)
        else:
            idx_train_retain.append(i)

    for i in range(len(testset)):
        if testset.targets[i] in class_to_forget:
            idx_test_forget.append(i)
        else:
            idx_test_retain.append(i)

    retain_train_set = Subset(trainset, idx_train_retain)
    forget_train_set = Subset(trainset,idx_train_forget)
    retain_test_set = Subset(testset, idx_test_retain)
    forget_test_set = Subset(testset, idx_test_forget)

    train_forget_loader = DataLoader(forget_train_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    train_retain_loader = DataLoader(retain_train_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_forget_loader = DataLoader(forget_test_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_retain_loader = DataLoader(retain_test_set, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)


    if args.data == "cifar100":
        class_num = 100
    else:
        class_num = 10
        
    idx_subset_descent = [0] * class_num
    #print("idx subset : ", idx_subset)
    idx_limit = [3] * class_num
    idx_plus = []

    for i in range(len(trainset)):
        if trainset.targets[i] in class_to_forget:
            pass
        else:
            if idx_subset_descent[trainset.targets[i]] == idx_limit[trainset.targets[i]]:
                pass
            else:
                idx_plus.append(i)
                idx_subset_descent[trainset.targets[i]] = idx_subset_descent[trainset.targets[i]] + 1

    print("idx subset descent : ", idx_subset_descent)


    idx_few = []
    idx_subset0 = 0
    idx_subset1 = 0
    for i in range(len(trainset)):
        if idx_subset0 == args.unlearn_samples and idx_subset1 == args.unlearn_samples:
            break
        else:
            if trainset.targets[i] == class_to_forget[0]:
                if idx_subset0 == args.unlearn_samples:
                    continue
                else:
                    idx_few.append(i)
                    idx_subset0 = idx_subset0 + 1
            if trainset.targets[i] == class_to_forget[1]:
                if idx_subset1 == args.unlearn_samples:
                    continue
                else:
                    idx_few.append(i)
                    idx_subset1 = idx_subset1 + 1
    
    #print("Samples used in LUV unlearn : ", idx_few)
    fewshot_train_descent = Subset(trainset, idx_plus)
    fewshot_train = Subset(trainset, idx_few)
    fewshot_descent_loader = DataLoader(fewshot_train_descent, batch_size = 129, shuffle=True, num_workers=args.num_workers) #batch size=129 for cifar100
    LUV_trainloader = DataLoader(fewshot_train, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)

    print("Finish Data Preprocessing")

    acc_full = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, full_test_loader, args)
    acc_remain = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_retain_loader, args)
    acc_forget = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_forget_loader, args)

    acc_remain_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_retain_loader, args)
    acc_forget_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_forget_loader, args)


    logger.info("Evaluate full VFL performance before unlearning.")
    logger.info("Evaluate full VFL performance on full dataset.")
    logger.info(f"Accuracy : {acc_full*100}%")
    logger.info("Full VFL performance on remaining dataset.")
    logger.info(f"Accuracy : {acc_remain*100}%")
    logger.info("Full VFL performance on forgetting dataset.")
    logger.info(f"Accuracy : {acc_forget*100}%\n")
   

    logger.info("Evaluate retrain VFL performance.")
    logger.info("Retrain VFL performance on remaining dataset.")
    logger.info(f"Accuracy : {acc_remain_retrain*100}%")
    logger.info("Retrain VFL performance on forgetting dataset.")
    logger.info(f"Accuracy : {acc_forget_retrain*100}%\n")
    

    logger.info("MIA on original model privacy before unlearning")
    metric = MIA(
        retain_loader=train_retain_loader,
        forget_loader=train_forget_loader,
        test_loader=full_test_loader,
        bottom_model_A=full_bottom_model_A,
        bottom_model_B = full_bottom_model_B,
        top_model=full_top_model,
        args=args
    )
    logger.info(metric*100)
    logger.info("\n")

    logger.info("MIA on retrain model privacy before unlearning")
    metric = MIA(
        retain_loader=train_retain_loader,
        forget_loader=train_forget_loader,
        test_loader=full_test_loader,
        bottom_model_A=retrain_bottom_model_A,
        bottom_model_B = retrain_bottom_model_B,
        top_model=retrain_top_model,
        args=args
    )
    logger.info(metric*100)
    logger.info("\n")

    unlearn_method = unlearn_2labels.get_unlearn_method(args.unlearn_method)

    criterion = nn.CrossEntropyLoss()

    #Start unlearning
    start_time = time.time()
    unlearn_method(LUV_trainloader, fewshot_descent_loader, full_bottom_model_A, full_bottom_model_B, full_top_model, class_to_forget, criterion, args)
    logger.info(f"Time for {args.unlearn_method} unlearning")
    end_time = time.time()
    duration = end_time - start_time
    duration = round(duration,2)
    logger.info("--- %s seconds ---" % (duration))
    logger.info("\n")



    acc_remain = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_retain_loader, args)
    acc_forget = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_forget_loader, args)

    logger.info("Evaluate full VFL performance on dataset after unlearning.")
    logger.info("Full VFL performance on remaining dataset.")
    logger.info(f"Accuracy : {acc_remain*100}%")
    logger.info("Full VFL performance on forgetting dataset.")
    logger.info(f"Accuracy : {acc_forget*100}%\n")
    


    logger.info("MIA on original model privacy after unlearning")
    metric = MIA(
        retain_loader=train_retain_loader,
        forget_loader=train_forget_loader,
        test_loader=full_test_loader,
        bottom_model_A=full_bottom_model_A,
        bottom_model_B = full_bottom_model_B,
        top_model=full_top_model,
        args=args
    )
    logger.info(metric*100)
    logger.info("\n")



if __name__ == "__main__":
    main()