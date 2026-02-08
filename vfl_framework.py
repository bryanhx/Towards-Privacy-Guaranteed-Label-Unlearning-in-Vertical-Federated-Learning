import logging

logger = logging.getLogger(f"VFU.{__name__}")
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import datasets, transforms


from dataset import get_dataset

from model.model_builder import get_model
from model import read_data_text

from utils import AvgMeter

from tqdm import tqdm

from copy import deepcopy
import math




class VFL(object):
    def __init__(self, args):
        self.args = args
        self.setup()
        self.test_acc_history = []
        self.best_bottom_model_A = None
        self.best_bottom_model_B = None
        self.best_top_model = None
        self.best_test_acc = 0.0


    def setup(self):
        
        if self.args.data.lower()=='yahoo':
            file_path = './data/yahoo_answers_csv/'
            train_set, _, _, _, _ = read_data_text.get_data(file_path, 5000)
            _, _, _, test_set, _ = read_data_text.get_data(file_path, 10)

        elif self.args.data.lower()=="mri":
            traindir = os.path.join('./data/Brain_MRI', 'Training')
            testdir = os.path.join('./data/Brain_MRI', 'Testing')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_set = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            test_set = datasets.ImageFolder(testdir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            
        elif self.args.data.lower()=="ct":
            traindir = os.path.join('./data/Chest CT_v2', 'train')
            testdir = os.path.join('./data/Chest CT_v2', 'test')
            normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])
            train_set = datasets.ImageFolder(
                        traindir,
                        transforms.Compose([
                            transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
            test_set = datasets.ImageFolder(testdir, transforms.Compose([
                            transforms.Resize(256),
                            transforms.CenterCrop(224),
                            transforms.ToTensor(),
                            normalize,
                        ]))

        else:
            dataset = get_dataset(self.args)
            train_set = deepcopy(dataset.trainset)
            test_set = deepcopy(dataset.testset)
        
        #Train a retrain model with remaining data
        if self.args.mode == 'retrain':
            idx_train_forget = []
            idx_train_retain = []
            idx_test_forget = []
            idx_test_retain = []
            if self.args.data == 'yahoo':
                class_to_forget = [self.args.unlearn_class]
                for i in range(len(train_set)):
                    if train_set.labels[i] in class_to_forget:
                        idx_train_forget.append(i)
                    else:
                        idx_train_retain.append(i)
    
                for i in range(len(test_set)):
                    if test_set.labels[i] in class_to_forget:
                        idx_test_forget.append(i)
                    else:
                        idx_test_retain.append(i)
            else:
                if self.args.unlearn_class_num == 1:
                    class_to_forget = [self.args.unlearn_class]
                elif self.args.unlearn_class_num == 2:
                    class_to_forget = [0,2]
                else: #4 unlearning classes. 
                    class_to_forget = [0,2,5,7]
                
                for i in range(len(train_set)):
                    if train_set.targets[i] in class_to_forget:
                        idx_train_forget.append(i)
                    else:
                        idx_train_retain.append(i)

                for i in range(len(test_set)):
                    if test_set.targets[i] in class_to_forget:
                        idx_test_forget.append(i)
                    else:
                        idx_test_retain.append(i)

            train_set = Subset(train_set, idx_train_retain)
            test_set = Subset(test_set, idx_test_retain)

        self.train_loader = DataLoader(train_set, batch_size=self.args.batch_size, shuffle=True,
                                       num_workers=self.args.num_workers)
        self.test_loader = DataLoader(test_set, batch_size=self.args.batch_size, shuffle=False,
                                      num_workers=self.args.num_workers)
        

        self.bottom_model_A = get_model(args=self.args)[0]
        self.bottom_model_B = get_model(args=self.args)[0]
        self.top_model = get_model(args=self.args)[1]
        self.bottom_model_A = self.bottom_model_A.to(self.args.device)
        self.bottom_model_B = self.bottom_model_B.to(self.args.device)
        self.top_model = self.top_model.to(self.args.device)
        logger.info(f"Bottom Model A:\n {self.bottom_model_A}")
        logger.info(f"Bottom Model B:\n {self.bottom_model_B}")
        logger.info(f"Top model:\n {self.top_model}")
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.train_acc_meter = AvgMeter()
        self.train_loss_meter = AvgMeter()
        self.test_acc_meter = AvgMeter()
        self.bottom_model_A_optimizer = torch.optim.SGD(self.bottom_model_A.parameters(),
                                                        self.args.optimizer_lr, 
                                                        momentum=self.args.momentum,
                                                        weight_decay=self.args.weight_decay)
        self.bottom_model_B_optimizer = torch.optim.SGD(self.bottom_model_B.parameters(),
                                                        self.args.optimizer_lr, 
                                                        momentum=self.args.momentum,
                                                        weight_decay=self.args.weight_decay)
        self.top_model_optimizer = torch.optim.SGD(self.top_model.parameters(),
                                                        self.args.optimizer_lr, 
                                                        momentum=self.args.momentum,
                                                        weight_decay=self.args.weight_decay)
        

    def train(self):
        self.bottom_model_A.train()
        self.bottom_model_B.train()
        self.top_model.train()
        self.train_acc_meter.reset()
        self.train_loss_meter.reset()
        for step, (batch_x, batch_y) in tqdm(enumerate(self.train_loader)):
            if self.args.data != 'yahoo':
                batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)
                #print("batch x : ", batch_x)
            if self.args.data == 'cifar10':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif self.args.data == 'cifar100':
                x_a = batch_x[:, :, :, 0:16]
                x_b = batch_x[:, :, :, 16:32]
            elif self.args.data == 'mnist':
                x_a = batch_x[:, :, :, 0:14]
                x_b = batch_x[:, :, :, 14:28]
            elif self.args.data == 'yahoo':
                for i in range(len(batch_x)):
                    batch_x[i] = batch_x[i].long().cuda()
                batch_y = batch_y[0].long().cuda()
                x_b = batch_x[1]
                x_a = batch_x[0]
            elif self.args.data == 'mri':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            elif self.args.data == 'ct':
                x_a = batch_x[:, :, :, 0:112]
                x_b = batch_x[:, :, :, 112:224]
            else:
                raise ValueError(f'No dataset named {self.args.data}!')
            
            output_tensor_bottom_model_a = self.bottom_model_A(x_a)
            output_tensor_bottom_model_b = self.bottom_model_B(x_b)

            input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            input_tensor_top_model_b = torch.tensor([], requires_grad=True)
            input_tensor_top_model_a.data = output_tensor_bottom_model_a.data
            input_tensor_top_model_b.data = output_tensor_bottom_model_b.data

            self.top_model_optimizer.zero_grad()
            output = self.top_model(input_tensor_top_model_a, input_tensor_top_model_b)

            loss = self.criterion(output, batch_y)
            acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
            self.train_acc_meter.update(acc.item(), batch_y.size(0))
            loss.backward()
            self.top_model_optimizer.step()
            self.train_loss_meter.update(loss.item(), output.size(0))

            grad_output_bottom_model_a = input_tensor_top_model_a.grad
            loss_bottom_A = torch.sum(grad_output_bottom_model_a * output_tensor_bottom_model_a)
            self.bottom_model_A_optimizer.zero_grad()
            loss_bottom_A.backward()
            self.bottom_model_A_optimizer.step()

            grad_output_bottom_model_b = input_tensor_top_model_b.grad
            loss_bottom_B = torch.sum(grad_output_bottom_model_b * output_tensor_bottom_model_b)
            self.bottom_model_B_optimizer.zero_grad()
            loss_bottom_B.backward()
            self.bottom_model_B_optimizer.step()


    def test(self):
        self.bottom_model_A.eval()
        self.bottom_model_B.eval()
        self.top_model.eval()
        self.test_acc_meter.reset()
        labels = []
        preds = []
        with torch.no_grad():
            for step, (batch_x, batch_y) in tqdm(enumerate(self.test_loader)):
                if self.args.data != 'yahoo':
                    batch_x, batch_y = batch_x.to(self.args.device), batch_y.to(self.args.device)

                if self.args.data == 'cifar10':
                    x_a = batch_x[:, :, :, 0:16]
                    x_b = batch_x[:, :, :, 16:32]
                elif self.args.data == 'cifar100':
                    x_a = batch_x[:, :, :, 0:16]
                    x_b = batch_x[:, :, :, 16:32]
                elif self.args.data == 'mnist':
                    x_a = batch_x[:, :, :, 0:14]
                    x_b = batch_x[:, :, :, 14:28]
                elif self.args.data == 'yahoo':
                    for i in range(len(batch_x)):
                        batch_x[i] = batch_x[i].long().cuda()
                    batch_y = batch_y[0].long().cuda()
                    x_b = batch_x[1]
                    x_a = batch_x[0]
                elif self.args.data == 'mri':
                    x_a = batch_x[:, :, :, 0:112]
                    x_b = batch_x[:, :, :, 112:224]
                elif self.args.data == 'ct':
                    x_a = batch_x[:, :, :, 0:112]
                    x_b = batch_x[:, :, :, 112:224]
                else:
                    raise ValueError(f'No dataset named {self.args.data}!')
                output_tensor_bottom_model_a = self.bottom_model_A(x_a)
                output_tensor_bottom_model_b = self.bottom_model_B(x_b)
                output = self.top_model(output_tensor_bottom_model_a, output_tensor_bottom_model_b)
                acc = torch.sum(torch.argmax(output, dim=-1) == batch_y) / batch_y.size(0)
                self.test_acc_meter.update(acc.item(), batch_y.size(0))
                self.test_acc_history.append(self.test_acc_meter.get())
            if self.test_acc_meter.get() > self.best_test_acc:
                self.best_test_acc = self.test_acc_meter.get()
                self.best_bottom_model_A = deepcopy(self.bottom_model_A)
                self.best_bottom_model_B = deepcopy(self.bottom_model_B)
                self.best_top_model = deepcopy(self.top_model)


    def adjust_lr(self):
        for param_group in self.bottom_model_B_optimizer.param_groups:
            param_group['lr'] *= self.args.gamma
        for param_group in self.bottom_model_A_optimizer.param_groups:
            param_group['lr'] *= self.args.gamma
        for param_group in self.top_model_optimizer.param_groups:
            param_group['lr'] *= self.args.gamma

    def save(self):
        save_path = self.args.outdir
        if self.args.mode == 'retrain':
            bottom_model_a_path = os.path.join(save_path, 'retrain_bottom_model_A.pt')
            torch.save(self.best_bottom_model_A.cpu(), bottom_model_a_path)
            bottom_model_b_path = os.path.join(save_path, 'retrain_bottom_model_B.pt')
            torch.save(self.best_bottom_model_B.cpu(), bottom_model_b_path)
            top_model_path = os.path.join(save_path, 'retrain_top_model.pt')
            torch.save(self.best_top_model.cpu(), top_model_path)
        else:
            bottom_model_a_path = os.path.join(save_path, 'full_bottom_model_A.pt')
            torch.save(self.best_bottom_model_A.cpu(), bottom_model_a_path)
            bottom_model_b_path = os.path.join(save_path, 'full_bottom_model_B.pt')
            torch.save(self.best_bottom_model_B.cpu(), bottom_model_b_path)
            top_model_path = os.path.join(save_path, 'full_top_model.pt')
            torch.save(self.best_top_model.cpu(), top_model_path)