from torchvision import transforms
import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch.nn.functional as F



def MIA(retain_loader, forget_loader, test_loader, bottom_model_A, bottom_model_B, top_model,args):
    X_f, Y_f, X_r, Y_r = get_membership_attack_data(retain_loader, forget_loader, test_loader, bottom_model_A, bottom_model_B, top_model,args)
    #clf = SVC(C=3,gamma='auto',kernel='rbf')
    clf = LogisticRegression(class_weight='balanced',solver='lbfgs')
    clf.fit(X_r, Y_r)
    results = clf.predict(X_f)
    return results.mean()

def get_membership_attack_data(retain_loader, forget_loader, test_loader, bottom_model_A, bottom_model_B, top_model,args):    
    retain_prob = collect_prob(retain_loader, bottom_model_A, bottom_model_B, top_model,args)
    forget_prob = collect_prob(forget_loader, bottom_model_A, bottom_model_B, top_model,args)
    test_prob = collect_prob(test_loader, bottom_model_A, bottom_model_B, top_model,args)
    
    X_r = torch.cat([entropy(retain_prob), entropy(test_prob)]).cpu().numpy().reshape(-1, 1)
    Y_r = np.concatenate([np.ones(len(retain_prob)), np.zeros(len(test_prob))])
    
    X_f = entropy(forget_prob).cpu().numpy().reshape(-1, 1)
    Y_f = np.concatenate([np.ones(len(forget_prob))])    
    return X_f, Y_f, X_r, Y_r


def entropy(p, dim = -1, keepdim = False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim=dim, keepdim=keepdim)

def collect_prob(data_loader, bottom_model_A, bottom_model_B, top_model,args):   
    #data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers = 40)
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            if args.data != 'yahoo':
                batch = [tensor.to(next(bottom_model_A.parameters()).device) for tensor in batch]
            data, target = batch
            if args.data == 'cifar10':
                x_a = data[:, :, :, 0:16]
                x_b = data[:, :, :, 16:32]
            elif args.data == 'mnist':
                x_a = data[:, :, :, 0:14]
                x_b = data[:, :, :, 14:28]
            elif args.data == 'cifar100':
                x_a = data[:, :, :, 0:16]
                x_b = data[:, :, :, 16:32]
            elif args.data == 'mri':
                x_a = data[:, :, :, 0:112]
                x_b = data[:, :, :, 112:224]
            elif args.data == 'ct':
                x_a = data[:, :, :, 0:112]
                x_b = data[:, :, :, 112:224]
            elif args.data == 'yahoo':
                for i in range(len(data)):
                    data[i] = data[i].long().cuda()
                target = target[0].long().cuda()
                x_b = data[1]
                x_a = data[0]
            else:
                raise ValueError(f'No dataset named {args.data}!')
            bottom_A_out = bottom_model_A(x_a)
            bottom_B_out = bottom_model_B(x_b)
            output = top_model(bottom_A_out, bottom_B_out)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)