import logging

logger = logging.getLogger(f"VFU.{__name__}")
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


from dataset import get_dataset

from model.model_builder import get_model

from utils import AvgMeter

from tqdm import tqdm
from torch.utils.data import Dataset
from skimage import io
from copy import deepcopy
import torchvision.models as models



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



class VFL_modelnet(object):
    def __init__(self, args):
        self.args = args
        self.setup()
        self.test_acc_history = []
        self.best_bottom_model_A = None
        self.best_bottom_model_B = None
        self.best_top_model = None
        self.best_test_acc = 0.0


    def setup(self):
        train_set = ModelNet(train=True)
        test_set = ModelNet(train=False)
        #Train a retrain model with remaining data
        if self.args.mode == 'retrain':
            class_to_forget = self.args.unlearn_class
            idx_train_forget = []
            idx_train_retain = []
            idx_test_forget = []
            idx_test_retain = []
            for i in range(len(train_set)):
                if train_set.label2[i] == class_to_forget:
                    idx_train_forget.append(i)
                else:
                    idx_train_retain.append(i)

            for i in range(len(test_set)):
                if test_set.label2[i] == class_to_forget:
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
        for step, (batch_x1, batch_x2, batch_y) in tqdm(enumerate(self.train_loader)):
            batch_x1, batch_x2, batch_y = batch_x1.to(self.args.device), batch_x2.to(self.args.device), batch_y.to(self.args.device)
    
            batch_x1 = batch_x1.float()
            batch_x2 = batch_x2.float()
            output_tensor_bottom_model_a = self.bottom_model_A(batch_x1)
            output_tensor_bottom_model_b = self.bottom_model_B(batch_x2)

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
            for step, (batch_x1, batch_x2, batch_y) in tqdm(enumerate(self.test_loader)):
                batch_x1, batch_x2, batch_y = batch_x1.to(self.args.device), batch_x2.to(self.args.device), batch_y.to(self.args.device)
        
                batch_x1 = batch_x1.float()
                batch_x2 = batch_x2.float()
                output_tensor_bottom_model_a = self.bottom_model_A(batch_x1)
                output_tensor_bottom_model_b = self.bottom_model_B(batch_x2)
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

