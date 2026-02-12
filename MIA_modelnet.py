from torchvision import transforms
import torch
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
import torch.nn.functional as F


def MIA(retain_loader, forget_loader, test_loader, bottom_model_A, bottom_model_B, top_model,args):
    print("MIA in ModelNet")
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

def collect_prob(data_loader, bottom_model_A, bottom_model_B,top_model,args):   
    #data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=1, shuffle=False, num_workers = 40)
    prob = []
    with torch.no_grad():
        for batch in data_loader:
            batch = [tensor.to(next(bottom_model_A.parameters()).device) for tensor in batch]
            batch_x1, batch_x2, batch_y = batch
            batch_x1 = batch_x1.float()
            batch_x2 = batch_x2.float()
            bottom_A_out = bottom_model_A(batch_x1)
            bottom_B_out = bottom_model_B(batch_x2)
            output = top_model(bottom_A_out, bottom_B_out)
            prob.append(F.softmax(output, dim=-1).data)
    return torch.cat(prob)