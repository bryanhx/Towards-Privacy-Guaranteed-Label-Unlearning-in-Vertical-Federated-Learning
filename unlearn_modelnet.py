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

import unlearn_modelnet

from configs import config
from utils import update_logger
from utils_modelnet import evaluate
from vfl_framework import VFL
from dataset import get_dataset
from copy import deepcopy
from torch.utils.data import DataLoader, Subset
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch import nn
import time
import numpy as np
import random
from MIA_modelnet import MIA
from skimage import io
logger = logging.getLogger(f"VFU.{__name__}")



class ModelNet(Dataset):
	def __init__(self, train = False, transforms = None):

		self.root = './data/modelnet40_2views_png'
		self.data = []
		self.v1 = []
		self.v2 = []
		self.labels = []
		self.label2 = []

		self.train_data = [file for file in os.listdir(self.root) if "train" in file]
		self.test_data = [file for file in os.listdir(self.root) if "test" in file]

		
		self.data_files = self.train_data if train else self.test_data

		
		self.load_data()



	def __len__(self):
		return len(self.v1)

	def __getitem__(self, idx):

		if torch.is_tensor(idx):
			idx = idx.tolist()

		v1 = self.v1[idx]
		image_v1 = io.imread(v1)

		v2 = self.v2[idx]
		image_v2 = io.imread(v2)

		label = self.label2[idx]

		
		image_v1 = image_v1.transpose((2, 0, 1))
		image_v2 = image_v2.transpose((2, 0, 1))
		label = torch.tensor(label)

		return torch.from_numpy(image_v1), torch.from_numpy(image_v2), label


	def load_data(self):
		temp = []
		temp_label = []
		n = 0
		for file in self.data_files:
			file_path = os.path.join(self.root, file)
			for label in os.listdir(file_path):
				temp_label.append(label)
				temp.append(label)

			temp_label.sort()
			for i in range(len(temp_label)):
				file_name = os.path.join(file_path, temp_label[i])
				temp1 = []
				temp2 = []
				filelist = []
				for file in os.listdir(file_name):
					filelist.append(file)

				filelist.sort()
				for file2 in range(len(filelist)):
					if filelist[file2][-7:-4] == "001":
						self.v1.append(os.path.join(file_name, filelist[file2]))
						temp1.append(os.path.join(file_name, filelist[file2]))
						
					if filelist[file2][-7:-4] == "002":
						self.v2.append(os.path.join(file_name, filelist[file2]))
						temp2.append(os.path.join(file_name, filelist[file2]))

				for i in range(len(temp1)):
					self.labels.append(label)
					self.label2.append(n)

					
				n = n + 1


def main():
    args = config()
    update_logger(args)
    args.device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    torch.backends.cudnn.deterministic=True
    
    full_bottom_model_A = torch.load("/scratch/taehongxi/LUV_Privacy/exp/LUV_resnet18_on_modelnet_lr0.001/_sub_exp_20260212115707/full_bottom_model_A.pt", weights_only=False)
    full_bottom_model_B = torch.load("/scratch/taehongxi/LUV_Privacy/exp/LUV_resnet18_on_modelnet_lr0.001/_sub_exp_20260212115707/full_bottom_model_B.pt", weights_only=False)
    full_top_model = torch.load("/scratch/taehongxi/LUV_Privacy/exp/LUV_resnet18_on_modelnet_lr0.001/_sub_exp_20260212115707/full_top_model.pt", weights_only=False)
    # retrain_bottom_model_A = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_modelnet_lr0.001/retrain3/retrain_bottom_model_A.pt", weights_only=False)
    # retrain_bottom_model_B = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_modelnet_lr0.001/retrain3/retrain_bottom_model_B.pt", weights_only=False)
    # retrain_top_model = torch.load("/home/user/bryanthx/exp/EE_resnet18_on_modelnet_lr0.001/retrain3/retrain_top_model.pt", weights_only=False)
    
    
    full_bottom_model_A = full_bottom_model_A.to(args.device)
    full_bottom_model_B = full_bottom_model_B.to(args.device)
    full_top_model = full_top_model.to(args.device)
    # retrain_bottom_model_A = retrain_bottom_model_A.to(args.device)
    # retrain_bottom_model_B = retrain_bottom_model_B.to(args.device)
    # retrain_top_model = retrain_top_model.to(args.device)
    
    
    trainset = ModelNet(train=True)
    testset = ModelNet(train=False)
    
    full_train_loader = DataLoader(trainset, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    full_test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    
    class_to_forget = args.unlearn_class
    idx_train_forget = []
    idx_train_retain = []
    idx_test_forget = []
    idx_test_retain = []
    for i in range(len(trainset)):
        if trainset.label2[i] == class_to_forget:
            idx_train_forget.append(i)
        else:
            idx_train_retain.append(i)

    for i in range(len(testset)):
        if testset.label2[i] == class_to_forget:
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
	
    idx_subset_descent = [0] * 40
    idx_limit = [3] * 40
    idx_plus = []
	
    for i in range(len(trainset)):
        #print("trainset targets : ", trainset.targets[i])
        if trainset.label2[i] == class_to_forget:
            #print("pass")
            pass
        else:
            #print("index descent b4 : ", idx_subset_descent)
            if idx_subset_descent[trainset.label2[i]] == idx_limit[trainset.label2[i]]:
                #print("pass idx")
                pass
            else:
                idx_subset_descent[trainset.label2[i]] = idx_subset_descent[trainset.label2[i]] + 1
                idx_plus.append(i)

    
    idx_few = []
    idx_subset = 0
    for i in range(len(forget_train_set)):
        if idx_subset == args.unlearn_samples:
            break
        else:
            idx_few.append(i)
            idx_subset = idx_subset + 1
    
    print("Samples used in LUV unlearn : ", idx_subset)
    fewshot_train_descent = Subset(trainset, idx_plus)
    fewshot_train = Subset(forget_train_set, idx_few)
    fewshot_descent_loader = DataLoader(fewshot_train_descent, batch_size = 32, shuffle=True, num_workers=args.num_workers) #batch size=129 for cifar100
    LUV_trainloader = DataLoader(fewshot_train, batch_size = args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    print("Finish Data Preprocessing")

    
    # acc_full = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, full_test_loader, args)
    # acc_remain = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_retain_loader, args)
    # acc_forget = evaluate(full_bottom_model_A, full_bottom_model_B, full_top_model, test_forget_loader, args)

    # acc_remain_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_retain_loader, args)
    # acc_forget_retrain = evaluate(retrain_bottom_model_A, retrain_bottom_model_B, retrain_top_model, test_forget_loader, args)

    
    # logger.info("Evaluate full VFL performance before unlearning.")
    # logger.info("Evaluate full VFL performance on full dataset.")
    # logger.info(f"Accuracy : {acc_full*100}%")
    # logger.info("Full VFL performance on remaining dataset.")
    # logger.info(f"Accuracy : {acc_remain*100}%")
    # logger.info("Full VFL performance on forgetting dataset.")
    # logger.info(f"Accuracy : {acc_forget*100}%\n")

    # logger.info("Evaluate retrain VFL performance.")
    # logger.info("Retrain VFL performance on remaining dataset.")
    # logger.info(f"Accuracy : {acc_remain_retrain*100}%")
    # logger.info("Retrain VFL performance on forgetting dataset.")
    # logger.info(f"Accuracy : {acc_forget_retrain*100}%\n")

    
    # start_time = time.time()
    # logger.info("MIA on original model privacy before unlearning")
    # metric = MIA(
    #     retain_loader=train_retain_loader,
    #     forget_loader=train_forget_loader,
    #     test_loader=full_test_loader,
    #     bottom_model_A=full_bottom_model_A,
    #     bottom_model_B = full_bottom_model_B,
    #     top_model=full_top_model,
    #     args=args
    # )
    # logger.info(metric*100)
    # logger.info("\n")
    # logger.info(f"Time for MIA")
    # end_time = time.time()
    # duration = end_time - start_time
    # duration = round(duration,2)
    # logger.info("--- %s seconds ---" % (duration))
    # logger.info("\n")

    # logger.info("MIA on retrain model privacy before unlearning")
    # metric = MIA(
    #     retain_loader=train_retain_loader,
    #     forget_loader=train_forget_loader,
    #     test_loader=full_test_loader,
    #     bottom_model_A=retrain_bottom_model_A,
    #     bottom_model_B = retrain_bottom_model_B,
    #     top_model=retrain_top_model,
    #     args=args
    # )
    # logger.info(metric*100)
    # logger.info("\n")
    
    unlearn_method = unlearn_modelnet.get_unlearn_method(args.unlearn_method)
    
    
    criterion = nn.CrossEntropyLoss()

    #Start unlearning
    start_time = time.time()
    unlearn_method(LUV_trainloader, fewshot_descent_loader, full_bottom_model_A, full_bottom_model_B, full_top_model, criterion, class_to_forget, args)
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

    
    logger.info("MIA on unlearn model privacy after unlearning")
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




    