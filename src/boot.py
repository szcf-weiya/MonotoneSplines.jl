import torch
import torch.nn as nn
import numpy as np
import tqdm as tqdm

class Model(nn.Module):
    def __init__(self, y, B, nhidden):
        super(Model, self).__init__()
        n = len(y)
        J = B.size()[1]
        self.MLP = nn.Sequential(
            nn.Linear(n, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, J)
        )
    def forward(self, y):
        beta_unsort = self.MLP(y)
        beta, indices = torch.sort(beta_unsort)
        return beta


# two different optimizers
# support lambda
def train_G(y, B, L, lam, K = 10, nepoch = 100, eta = 0.001, sigma = 1):
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(y).to(device)
    B = torch.from_numpy(B).to(device)
    L = torch.from_numpy(L).to(device)
    n, J = B.size()
    model = Model(y, B, 1000).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr = eta / K, amsgrad = True)
    opt2 = torch.optim.Adam(model.parameters(), lr = eta, amsgrad = True)
    loss_fn = nn.functional.mse_loss
    # just realized that pytorch also did not do sort in batch
    LOSS = torch.zeros(nepoch, 2)
    pbar = tqdm.trange(nepoch, desc="Training G")
    # for epoch in range(nepoch):
    for epoch in pbar:
        ## first step
        beta = model(y) # 1xJ
        ypred = torch.matmul(beta, B.t())
        # not directly use sum on the penalty loss, so pay attention whether the loss has scaled by n
        # ...........................................................1xJ x JxJ
        # note that it should be L' * beta, so TODO: double check whether a transpose is necessary
        loss1_fit = loss_fn(ypred, y)
        loss1 = loss1_fit + lam * torch.square(torch.matmul(beta, L)).mean() 
        opt1.zero_grad()
        loss1.backward()
        opt1.step()
        #
        sigma = np.sqrt(loss1_fit.cpu().detach().numpy())
        ## second step
        epsilons = torch.randn((K, n)).to(device) * sigma
        #        1xn      +      Kxn
        ytrain = ypred + epsilons 
        betas = model(ytrain) # K x J
        yspred = torch.matmul(betas, B.t()) # nxJ x JxK
        # ...............................................................KxJ x JxJ
        loss2 = loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() 
        #
        opt2.zero_grad()
        loss2.backward()
        opt2.step()
        #
        LOSS[epoch, 0] = loss1.item()
        LOSS[epoch, 1] = loss2.item()
        # print(f"epoch = {epoch}, loss = {LOSS[epoch,]}, sigma = {sigma}")
        pbar.set_postfix(epoch = epoch, loss = LOSS[epoch,], sigma = sigma)
    #
    #
    G = lambda y: model(torch.from_numpy(y).to(device)).cpu().detach().numpy() # support y is Float32
    return G, LOSS.cpu().detach().numpy()



