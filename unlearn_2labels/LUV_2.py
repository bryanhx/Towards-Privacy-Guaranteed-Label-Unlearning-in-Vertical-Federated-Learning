from utils import AvgMeter
import torch
from torch import nn
import math
from torch.autograd import grad
import numpy as np


def LUV_2(LUV_trainloader, fewshot_descent_loader, bottom_model_A, bottom_model_B, top_model, class_to_forget, criterion, args):
    print("unlearn method : ", args.unlearn_method)
    bottom_model_A.train()
    bottom_model_B.train()
    top_model.train()

    bottom_model_A_optimizer = torch.optim.SGD(
                                    bottom_model_A.parameters(), 
                                    args.unlearn_lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
            
    bottom_model_B_optimizer = torch.optim.SGD(
                                    bottom_model_B.parameters(),
                                    args.unlearn_lr, 
                                    momentum=args.momentum, 
                                    weight_decay=args.weight_decay)
    top_model_optimizer = torch.optim.SGD(
                        top_model.parameters(),
                        args.unlearn_lr, 
                        momentum=args.momentum, 
                        weight_decay=args.weight_decay)
    
    for epoch in range(args.unlearn_epochs):
        for i, (images, labels) in enumerate(LUV_trainloader):
            images = images.cuda()
            labels = labels.cuda()
            list_0 = (labels == 0).nonzero(as_tuple=True)
            list_1 = (labels == 2).nonzero(as_tuple=True)
            list_0 = list_0[0].cpu().numpy()
            list_1 = list_1[0].cpu().numpy()
            if args.data == 'cifar10':
                x_a = images[:, :, :, 0:16]
                x_b = images[:, :, :, 16:32]
            elif args.data == 'cifar100':
                x_a = images[:, :, :, 0:16]
                x_b = images[:, :, :, 16:32]
            else:
                raise ValueError(f'No dataset named {args.data}!')
            top_model_optimizer.zero_grad()

            output_tensor_bottom_model_a = bottom_model_A(x_a)
            output_tensor_bottom_model_b = bottom_model_B(x_b)

            list_1_a = []
            list_1_b = []
            list_2_a = []
            list_2_b = []

            for i in range(output_tensor_bottom_model_a.shape[0]):
                if labels[i] == class_to_forget[0]:
                    list_1_a.append(output_tensor_bottom_model_a[i])
                    list_1_b.append(output_tensor_bottom_model_b[i])
                else:
                    list_2_a.append(output_tensor_bottom_model_a[i])
                    list_2_b.append(output_tensor_bottom_model_b[i])

            list_1_a = torch.stack(list_1_a, dim=0)
            list_1_b = torch.stack(list_1_b, dim=0)
            list_2_a = torch.stack(list_2_a, dim=0)
            list_2_b = torch.stack(list_2_b, dim=0)
            
            mixup = [0.25, 0.5, 0.75]
            for i in range(list_1_a.shape[0]):
                x1_bottom_a = list_1_a[i]
                x1_bottom_b = list_1_b[i]
                if i == list_1_a.shape[0]-2:
                    mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                    mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                    front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                    front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                    back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                    back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                    temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                    temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                    bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                    bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)

                else:
                    for x in range(i+1, list_1_a.shape[0]):
                        x2_bottom_a = list_1_a[x]
                        x2_bottom_b = list_1_b[x]
                        mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                        mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                        front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                        front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                        back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                        back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                        if x == i+1:
                            if i == 0:
                                bottom_a_embedding_exp = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                                bottom_b_embedding_exp = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                            else:
                                temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                                temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                                bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                                bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)
                        else:
                            temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                            temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                            bottom_a_embedding_exp = torch.cat((bottom_a_embedding_exp,temperory_bottom_a),0)
                            bottom_b_embedding_exp = torch.cat((bottom_b_embedding_exp, temperory_bottom_b),0)


            for j in range(list_2_a.shape[0]):
                x1_bottom_a = list_2_a[j]
                x1_bottom_b = list_2_b[j]
                if j == list_2_a.shape[0]-2:
                    mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                    mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                    front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                    front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                    back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                    back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                    temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                    temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                    bottom_a_embedding_exp2 = torch.cat((bottom_a_embedding_exp2,temperory_bottom_a),0)
                    bottom_b_embedding_exp2 = torch.cat((bottom_b_embedding_exp2, temperory_bottom_b),0)

                else:
                    for y in range(j+1, list_2_a.shape[0]):
                        x2_bottom_a = list_2_a[y]
                        x2_bottom_b = list_2_b[y]
                        mid_bottom_a = mixup[1]*x1_bottom_a + (1-mixup[1])*x2_bottom_a
                        mid_bottom_b = mixup[1]*x1_bottom_b + (1-mixup[1])*x2_bottom_b
                        front_mid_bottom_a = mixup[0]*x1_bottom_a + (1-mixup[0])*x2_bottom_a
                        front_mid_bottom_b = mixup[0]*x1_bottom_b + (1-mixup[0])*x2_bottom_b
                        back_mid_bottom_a = mixup[2]*x1_bottom_a + (1-mixup[2])*x2_bottom_a
                        back_mid_bottom_b = mixup[2]*x1_bottom_b + (1-mixup[2])*x2_bottom_b
                        if y == j+1:
                            if j == 0:
                                bottom_a_embedding_exp2 = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                                bottom_b_embedding_exp2 = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                            else:
                                temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                                temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                                bottom_a_embedding_exp2 = torch.cat((bottom_a_embedding_exp2,temperory_bottom_a),0)
                                bottom_b_embedding_exp2 = torch.cat((bottom_b_embedding_exp2, temperory_bottom_b),0)
                        else:
                            temperory_bottom_a = torch.stack((front_mid_bottom_a,mid_bottom_a,back_mid_bottom_a))
                            temperory_bottom_b = torch.stack((front_mid_bottom_b,mid_bottom_b,back_mid_bottom_b))
                            bottom_a_embedding_exp2 = torch.cat((bottom_a_embedding_exp2,temperory_bottom_a),0)
                            bottom_b_embedding_exp2 = torch.cat((bottom_b_embedding_exp2, temperory_bottom_b),0)

                                
            label1 = torch.full((bottom_a_embedding_exp.shape[0],), class_to_forget[0], dtype=int)
            label2 = torch.full((bottom_a_embedding_exp2.shape[0],), class_to_forget[1], dtype=int)
            bottom_a_embedding_exp_combine = torch.cat((bottom_a_embedding_exp, bottom_a_embedding_exp2),0)
            bottom_b_embedding_exp_combine = torch.cat((bottom_b_embedding_exp, bottom_b_embedding_exp2),0)
            label = torch.cat((label1,label2),0)
            indices = torch.randperm(bottom_a_embedding_exp_combine.size()[0])
            bottom_a_embedding_exp_combine = bottom_a_embedding_exp_combine[indices]
            bottom_b_embedding_exp_combine = bottom_b_embedding_exp_combine[indices]
            label = label[indices]
            label_copy = label[indices]

            input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            input_tensor_top_model_a.data = bottom_a_embedding_exp_combine.data
            input_tensor_top_model_b = torch.tensor([], requires_grad=True)
            input_tensor_top_model_b.data = bottom_b_embedding_exp_combine.data

            forward_bottom_a = []
            forward_bottom_b = []
            label_list = []

            for w in range(math.ceil(bottom_a_embedding_exp_combine.shape[0]/args.batch_size)):
                if w == 0:
                    forward_bottom_a.append(input_tensor_top_model_a[0:args.batch_size])
                    forward_bottom_b.append(input_tensor_top_model_b[0:args.batch_size])
                    label_list.append(label[0:args.batch_size])
                else:
                    forward_bottom_a.append(input_tensor_top_model_a[(w*args.batch_size):((w+1)*args.batch_size)])
                    forward_bottom_b.append(input_tensor_top_model_b[(w*args.batch_size):((w+1)*args.batch_size)])
                    label_list.append(label[(w*args.batch_size):((w+1)*args.batch_size)])


            grad_output_bottom_model_a = torch.tensor([])
            grad_output_bottom_model_a = grad_output_bottom_model_a.to(args.device)
            grad_output_bottom_model_b = torch.tensor([])
            grad_output_bottom_model_b = grad_output_bottom_model_b.to(args.device)


            for z in range(len(forward_bottom_a)):
                forward_bottom_a[z] = forward_bottom_a[z].detach()
                forward_bottom_a[z].requires_grad=True
                forward_bottom_b[z] = forward_bottom_b[z].detach()
                forward_bottom_b[z].requires_grad=True
                outputs = top_model(forward_bottom_a[z], forward_bottom_b[z])
                label = label_list[z]
                label = label.to('cuda')
                loss = -criterion(outputs, label)
                loss.backward()
                top_model_optimizer.step()

                grad_output_bottom_model_a = torch.cat((grad_output_bottom_model_a, forward_bottom_a[z].grad),0)
                grad_output_bottom_model_b = torch.cat((grad_output_bottom_model_b, forward_bottom_b[z].grad),0)

            list_1_a = []
            list_1_b = []
            list_2_a = []
            list_2_b = []
            for i in range(grad_output_bottom_model_a.shape[0]):
                if label_copy[i] == class_to_forget[0]:
                    list_1_a.append(grad_output_bottom_model_a[i])
                    list_1_b.append(grad_output_bottom_model_b[i])
                else:
                    list_2_a.append(grad_output_bottom_model_a[i])
                    list_2_b.append(grad_output_bottom_model_b[i])

            list_1_a = torch.stack(list_1_a, dim=0)
            list_1_b = torch.stack(list_1_b, dim=0)
            list_2_a = torch.stack(list_2_a, dim=0)
            list_2_b = torch.stack(list_2_b, dim=0)


            n = math.ceil(list_1_a.shape[0]/len(list_0))
            m = math.ceil(list_2_a.shape[0]/len(list_1))
            grad_1_a = torch.Tensor([])
            grad_1_a = grad_1_a.to(args.device)
            grad_2_a = torch.Tensor([])
            grad_2_a = grad_2_a.to(args.device)
            for k in range(len(list_0)):
                if k == 0:
                    grad_1_a = torch.cat((grad_1_a,(torch.mean(list_1_a[k:n], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_1_a = torch.cat((grad_1_a,(torch.mean(list_1_a[(k*n):((k+1)*n)], axis=0)).unsqueeze(0)), 0)

            for l in range(len(list_1)):
                if l == 0:
                    grad_2_a = torch.cat((grad_2_a,(torch.mean(list_2_a[l:m], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_2_a = torch.cat((grad_2_a,(torch.mean(list_2_a[(l*m):((l+1)*m)], axis=0)).unsqueeze(0)), 0)


            grad_a = torch.Tensor([])
            grad_a = grad_a.to(args.device)
            n = 0
            m = 0
            for a in range(labels.shape[0]):
                if labels[a] == class_to_forget[0]:
                    grad_a = torch.cat((grad_a, grad_1_a[n].unsqueeze(0)),0)
                    n = n + 1
                else:
                    grad_a = torch.cat((grad_a, grad_2_a[m].unsqueeze(0)),0)
                    m = m + 1

            loss_bottom_A = torch.sum(grad_a * output_tensor_bottom_model_a)
            bottom_model_A_optimizer.zero_grad()
            loss_bottom_A.backward()
            bottom_model_A_optimizer.step()
            


            n = math.ceil(list_1_b.shape[0]/len(list_0))
            m = math.ceil(list_2_b.shape[0]/len(list_1))
            grad_1_b = torch.Tensor([])
            grad_1_b = grad_1_b.to(args.device)
            grad_2_b = torch.Tensor([])
            grad_2_b = grad_2_b.to(args.device)
            k=0
            l=0
            for k in range(len(list_0)):
                if k == 0:
                    grad_1_b = torch.cat((grad_1_b,(torch.mean(list_1_b[k:n], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_1_b = torch.cat((grad_1_b,(torch.mean(list_1_b[(k*n):((k+1)*n)], axis=0)).unsqueeze(0)), 0)

            for l in range(len(list_1)):
                if l == 0:
                    grad_2_b = torch.cat((grad_2_b,(torch.mean(list_2_b[l:m], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_2_b = torch.cat((grad_2_b,(torch.mean(list_2_b[(l*m):((l+1)*m)], axis=0)).unsqueeze(0)), 0)

            grad_b = torch.Tensor([])
            grad_b = grad_b.to(args.device)
            n = 0
            m = 0
            for a in range(labels.shape[0]):
                if labels[a] == class_to_forget[0]:
                    grad_b = torch.cat((grad_b, grad_1_b[n].unsqueeze(0)),0)
                    n = n + 1
                else:
                    grad_b = torch.cat((grad_b, grad_2_b[m].unsqueeze(0)),0)
                    m = m + 1

            loss_bottom_B = torch.sum(grad_b * output_tensor_bottom_model_b)
            bottom_model_B_optimizer.zero_grad()
            loss_bottom_B.backward()
            bottom_model_B_optimizer.step()

        alphaa=1
        for i, (images, labels) in enumerate(fewshot_descent_loader):
            images = images.cuda()
            labels = labels.cuda()
            if args.data == 'cifar10':
                x_a = images[:, :, :, 0:16]
                x_b = images[:, :, :, 16:32]
            elif args.data == 'cifar100':
                x_a = images[:, :, :, 0:16]
                x_b = images[:, :, :, 16:32]
            else:
                raise ValueError(f'No dataset named {args.data}!')
            top_model_optimizer.zero_grad()

            #print("labels : ", labels)

            output_tensor_bottom_model_a = bottom_model_A(x_a)
            output_tensor_bottom_model_b = bottom_model_B(x_b)

            emb_descent_a, emb_descent_b = mixup_embeddings(output_tensor_bottom_model_a, output_tensor_bottom_model_b, labels, 1)

            input_tensor_top_model_a_descent = torch.tensor([], requires_grad=True)
            input_tensor_top_model_a_descent.data = emb_descent_a.data
            input_tensor_top_model_b_descent = torch.tensor([], requires_grad=True)
            input_tensor_top_model_b_descent.data = emb_descent_b.data

            output_descent = top_model(input_tensor_top_model_a_descent, input_tensor_top_model_b_descent)
            loss_descent = alphaa * criterion(output_descent, labels)
            
            top_model_optimizer.zero_grad()
            loss_descent.backward()
            top_model_optimizer.step()

            grad_descent_a = input_tensor_top_model_a_descent.grad
            grad_descent_b = input_tensor_top_model_b_descent.grad

            loss_bottom_A = torch.sum(grad_descent_a * output_tensor_bottom_model_a)
            bottom_model_A_optimizer.zero_grad()
            loss_bottom_A.backward()
            bottom_model_A_optimizer.step()

            loss_bottom_B = torch.sum(grad_descent_b * output_tensor_bottom_model_b)
            bottom_model_B_optimizer.zero_grad()
            loss_bottom_B.backward()
            bottom_model_B_optimizer.step()




def mixup_embeddings(embeddings_a, embeddings_b, targets, alpha=1.0):
    """Performs manifold mixup on a batch of embeddings."""
    # if alpha > 0:
    #     lam = torch.distributions.Beta(alpha, alpha).sample().item()
    # else:
    #     lam = 1

    mixed_embeddings_a = embeddings_a.clone()
    mixed_embeddings_b = embeddings_b.clone()

    # print("targets : ", targets)
    # print("targets unique : ", targets.unique())
    for class_label in targets.unique():
        # Get indices of the current class
        idx = (targets == class_label).nonzero(as_tuple=True)[0]
        #print("idx : ", idx)
        if len(idx) < 2:
            continue  # Cannot mix if only one sample of a class

         # Shuffle indices within the class
        perm = idx[torch.randperm(len(idx))]
        #print("perm : ", perm)

        #Sample lambda from Beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample((len(idx),)).to(embeddings_a.device)
        lam = lam.view(-1, 1)
        
        # print("lam : ", lam)

        # Perform mixup
        emb1_a = embeddings_a[idx]
        emb2_a = embeddings_a[perm]
        emb1_b = embeddings_b[idx]
        emb2_b = embeddings_b[perm]

        mixed_a = lam * emb1_a + (1 - lam) * emb2_a
        mixed_b = lam * emb1_b + (1 - lam) * emb2_b
        
        mixed_embeddings_a[idx] = mixed_a
        mixed_embeddings_b[idx] = mixed_b

    return mixed_embeddings_a, mixed_embeddings_b

   

    
    
