from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

def pgd_attack(model,
               X,
               y,
               epsilon=0.031,
               num_steps=10,
               step_size=2/255,
               random = True
               ):
    X, y = Variable(X, requires_grad=True), Variable(y)
    out = model(X)
    loss_nat = nn.CrossEntropyLoss(reduction='sum')(out, y)
    acc = (out.data.max(1)[1] == y.data).sum().item()
    if random:
        random_noise = torch.FloatTensor(*X.shape).uniform_(-epsilon, epsilon).to(X.device)
        X_pgd = Variable(torch.clamp(X.data + random_noise, 0.0, 1.0), requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss.backward()
        eta = step_size * X_pgd.grad.data.sign()
        eta += X_pgd.data - X.data
        eta = torch.clamp(eta, -epsilon, epsilon)
        X_pgd = Variable(torch.clamp(X.data + eta, 0, 1.0), requires_grad=True)
    out_pgd = model(X_pgd)
    loss_pgd = nn.CrossEntropyLoss(reduction='sum')(out_pgd, y)
    acc_pgd = (out_pgd.data.max(1)[1] ==y.data).sum().item()
    #print('err pgd (white-box): ', err_pgd)
    return loss_nat, loss_pgd, acc, acc_pgd

