from utils import AvgMeter
import torch
from torch import nn
import math
from torch.autograd import grad
import numpy as np



def LUV_modelnet(LUV_trainloader, fewshot_descent_loader, bottom_model_A, bottom_model_B, top_model, criterion, class_to_forget, args):
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
    epochs = args.unlearn_epochs

    for epoch in range(epochs):
        for step, (batch_x1, batch_x2, batch_y) in enumerate(LUV_trainloader):
            batch_x1, batch_x2, batch_y = batch_x1.to(args.device), batch_x2.to(args.device), batch_y.to(args.device)
        
            batch_x1 = batch_x1.float()
            batch_x2 = batch_x2.float()

            top_model_optimizer.zero_grad()

            output_tensor_bottom_model_a = bottom_model_A(batch_x1)
            output_tensor_bottom_model_b = bottom_model_B(batch_x2)

            
            mixup = [0.25, 0.5, 0.75]
            for i in range(output_tensor_bottom_model_a.shape[0]):
                x1_bottom_a = output_tensor_bottom_model_a[i]
                x1_bottom_b = output_tensor_bottom_model_b[i]
                if i == output_tensor_bottom_model_a.shape[0]-2:
                    #x2_bottom_a = output_tensor_bottom_model_a[i+1]
                    #x2_bottom_b = output_tensor_bottom_model_b[i+1]
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
                    for x in range(i+1, output_tensor_bottom_model_a.shape[0]):
                        x2_bottom_a = output_tensor_bottom_model_a[x]
                        x2_bottom_b = output_tensor_bottom_model_b[x]
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

            input_tensor_top_model_a = torch.tensor([], requires_grad=True)
            #output_tensor_bottom_model_a.data = bottom_a_embedding_exp.data
            input_tensor_top_model_a.data = bottom_a_embedding_exp.data
            input_tensor_top_model_b = torch.tensor([], requires_grad=True)
            #output_tensor_bottom_model_b.data = bottom_b_embedding_exp.data
            input_tensor_top_model_b.data = bottom_b_embedding_exp.data

            shape_1 = input_tensor_top_model_a.shape[0]
            # print("output bottom model a shape after augment : ", input_tensor_top_model_a.shape)
            # print("output active model b shape after augment : ", output_tensor_bottom_model_a.shape)


            forward_bottom_a = []
            forward_bottom_b = []

            for y in range(math.ceil(bottom_a_embedding_exp.shape[0]/args.batch_size)):
                if y == 0:
                    forward_bottom_a.append(input_tensor_top_model_a[0:args.batch_size])
                    forward_bottom_b.append(input_tensor_top_model_b[0:args.batch_size])
                else:
                    forward_bottom_a.append(input_tensor_top_model_a[(y*args.batch_size):((y+1)*args.batch_size)])
                    forward_bottom_b.append(input_tensor_top_model_b[(y*args.batch_size):((y+1)*args.batch_size)])


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
                label = torch.full((outputs.shape[0],), args.unlearn_class, dtype=int)
                label = label.to('cuda')
                loss = -criterion(outputs, label)
                loss.backward()
                top_model_optimizer.step()

                grad_output_bottom_model_a = torch.cat((grad_output_bottom_model_a, forward_bottom_a[z].grad),0)
                grad_output_bottom_model_b = torch.cat((grad_output_bottom_model_b, forward_bottom_b[z].grad),0)


            
            n = math.ceil(grad_output_bottom_model_a.shape[0]/output_tensor_bottom_model_a.shape[0])
            grad_a = torch.tensor([])
            grad_a = grad_a.to(args.device)
            for m in range(output_tensor_bottom_model_a.shape[0]):
                if m == 0:
                    grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_model_a[m:n], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_a = torch.cat((grad_a,(torch.mean(grad_output_bottom_model_a[(m*n):((m+1)*n)], axis=0)).unsqueeze(0)), 0)
            loss_bottom_A = torch.sum(grad_a * output_tensor_bottom_model_a)
            bottom_model_A_optimizer.zero_grad()
            loss_bottom_A.backward()
            bottom_model_A_optimizer.step()

            p = math.ceil(grad_output_bottom_model_b.shape[0]/output_tensor_bottom_model_b.shape[0])
            grad_b = torch.tensor([])
            grad_b = grad_b.to(args.device)
            for q in range(output_tensor_bottom_model_b.shape[0]):
                if q == 0:
                    grad_b = torch.cat((grad_b,(torch.mean(grad_output_bottom_model_b[q:p], axis=0)).unsqueeze(0)), 0)
                else:
                    grad_b = torch.cat((grad_b,(torch.mean(grad_output_bottom_model_b[(q*p):((q+1)*p)], axis=0)).unsqueeze(0)), 0)
            loss_bottom_B = torch.sum(grad_b * output_tensor_bottom_model_b)
            bottom_model_B_optimizer.zero_grad()
            loss_bottom_B.backward()
            bottom_model_B_optimizer.step()


        alphaa=1
        for i, (batch_x1, batch_x2, batch_y) in enumerate(fewshot_descent_loader):
            batch_x1, batch_x2, batch_y = batch_x1.to(args.device), batch_x2.to(args.device), batch_y.to(args.device)
            #print("label : ", batch_y)
        
            batch_x1 = batch_x1.float()
            batch_x2 = batch_x2.float()

            top_model_optimizer.zero_grad()

            output_tensor_bottom_model_a = bottom_model_A(batch_x1)
            output_tensor_bottom_model_b = bottom_model_B(batch_x2)

            emb_descent_a, emb_descent_b = mixup_embeddings(output_tensor_bottom_model_a, output_tensor_bottom_model_b, batch_y, 1)

            input_tensor_top_model_a_descent = torch.tensor([], requires_grad=True)
            input_tensor_top_model_a_descent.data = emb_descent_a.data
            input_tensor_top_model_b_descent = torch.tensor([], requires_grad=True)
            input_tensor_top_model_b_descent.data = emb_descent_b.data

            output_descent = top_model(input_tensor_top_model_a_descent, input_tensor_top_model_b_descent)
            loss_descent = alphaa * criterion(output_descent, batch_y)
            
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

    #print("mix embedding a : ", mixed_embeddings_a)


    for class_label in targets.unique():
        # Get indices of the current class
        idx = (targets == class_label).nonzero(as_tuple=True)[0]
        if len(idx) < 2:
            continue  # Cannot mix if only one sample of a class

         # Shuffle indices within the class
        perm = idx[torch.randperm(len(idx))]

        #Sample lambda from Beta distribution
        lam = torch.distributions.Beta(alpha, alpha).sample((len(idx),)).to(embeddings_a.device)
        lam = lam.view(-1, 1)
        

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

        
        
