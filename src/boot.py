import torch
import torch.nn as nn
import numpy as np
import tqdm as tqdm

class Model(nn.Module):
    def __init__(self, n, J, nhidden):
        super(Model, self).__init__()
        # if y is a vector, suppose it has been added 1 dim such that the batch size = 1.
        # n = len(y[0])
        # n, J = B.size()
        self.MLP = nn.Sequential(
            nn.Linear(n, nhidden),
            # nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            # nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            # nn.BatchNorm1d(nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, J)
        )
    def forward(self, y):
        beta_unsort = self.MLP(y)
        beta, indices = torch.sort(beta_unsort)
        return beta


# two different optimizers
# support lambda
def train_G(y, B, L, lam, K = 10, nepoch = 100, nhidden = 1000, eta = 0.001, eta0 = 0.0001, gamma = 0.9, sigma = 1, amsgrad = False, decay_step = 1000, max_norm = 2.0, clip_ratio = 1.0):
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(y[None, :]).to(device)
    B = torch.from_numpy(B).to(device)
    L = torch.from_numpy(L).to(device)
    n, J = B.size()
    model = Model(n, J, nhidden).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr = eta0, amsgrad = amsgrad)
    opt2 = torch.optim.Adam(model.parameters(), lr = eta, amsgrad = amsgrad)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, gamma = gamma, step_size = decay_step)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, gamma = gamma, step_size = decay_step)
    loss_fn = nn.functional.mse_loss
    # just realized that pytorch also did not do sort in batch
    LOSS = torch.zeros(nepoch, 2)
    pbar = tqdm.trange(nepoch, desc="Training G")
    # for epoch in range(nepoch):
    for epoch in pbar:
        ## first step
        beta = model(y) # 1xJ (actually just array vector if not expanding dim)
        ypred = torch.matmul(beta, B.t())
        # not directly use sum on the penalty loss, so pay attention the loss has scaled by J
        # ...........................................................1xJ x JxJ
        # note that it should be L' * beta, so TODO: double check whether a transpose is necessary
        loss1_fit = loss_fn(ypred, y)
        loss1 = loss1_fit + lam * torch.square(torch.matmul(beta, L)).mean() * J
        opt1.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        # nn.utils.clip_grad_value_(model.parameters(), max_norm)
        opt1.step()
        sch1.step()
        #
        sigma = torch.std(ypred.detach() - y, unbiased = True)
        ## second step
        epsilons = torch.randn((K, n)).to(device) * sigma
        #        1xn      +      Kxn
        # https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics
        ytrain = ypred.detach() + epsilons 
        betas = model(ytrain) # K x J
        yspred = torch.matmul(betas, B.t()) # KxJ x Jxn
        # ...............................................................KxJ x JxJ
        loss2 = loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() 
        #
        opt2.zero_grad()
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm * clip_ratio)
        # nn.utils.clip_grad_value_(model.parameters(), max_norm)
        opt2.step()
        sch2.step()
        #
        LOSS[epoch, 0] = loss1.item()
        LOSS[epoch, 1] = loss2.item()
        # print(f"epoch = {epoch}, loss = {LOSS[epoch,]}, sigma = {sigma}")
        pbar.set_postfix(epoch = epoch, loss = LOSS[epoch,], sigma = sigma, lr1 = sch1.get_last_lr())
    #
    #
    G = lambda y: model(torch.from_numpy(y).to(device)).cpu().detach().numpy() # support y is Float32
    return G, LOSS.cpu().detach().numpy()



