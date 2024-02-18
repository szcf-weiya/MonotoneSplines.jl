import torch
import torch.nn as nn
import numpy as np
import tqdm as tqdm
import pickle
import torchsort
import copy
from pytorchtools import EarlyStopping

class Model(nn.Module):
    def __init__(self, n, J, nhidden, depth = 2, use_torchsort = False, sort_reg_strength = 0.1):
        super(Model, self).__init__()
        # if y is a vector, suppose it has been added 1 dim such that the batch size = 1.
        # n = len(y[0])
        # n, J = B.size()
        self.use_torchsort = use_torchsort
        self.sort_reg_strength = sort_reg_strength
        self.fin = nn.Linear(n, nhidden)
        self.linears = nn.ModuleList([nn.Linear(nhidden, nhidden) for i in range(depth)])
        self.fout = nn.Linear(nhidden, J)
        # activation = nn.LeakyReLU()
        # activation = nn.PReLU()
        self.activation = nn.GELU()
    def forward(self, y):
        y = self.activation(self.fin(y))
        for i, l in enumerate(self.linears):
            y = self.activation(l(y))
        beta_unsort = self.fout(y)
        if self.use_torchsort:
            beta = torchsort.soft_sort(beta_unsort, regularization_strength=self.sort_reg_strength)
        else:
            beta, indices = torch.sort(beta_unsort)
        return beta

def train_G_lambda(y, B, L, K = 10, K0 = 10, 
                    eta = 0.001, eta0 = 0.0001, 
                    nhidden = 1000, depth = 2, 
                    nepoch = 100, nepoch0 = 100,
                    niter_per_epoch = 100,
                    gamma = 0.9, decay_step = 5, # TODO: how to proper set schedule? or just discard?
                    lam_lo = 1e-9, lam_up = 1e-4, use_torchsort = False, 
                    sort_reg_strength = 0.1, gpu_id = 0, 
                    patience = 100, patience0 = 100, disable_early_stopping = True, # TODO: early stopping  
                    eval_sigma_adaptive = False, # if False, use `model0` to evaluate sigma
                    model_file = "model_G.pt", 
                    step2_use_tensor = True, amsgrad = True, # no need to modified,
                    lams_opt_train = None, lams_opt_val = None, # each lam corresponds to a beta (dim: N)
                    betas_opt_train = None, betas_opt_val = None, # evaluate the loss between the OPT solution and GpBS solution here (dim NxJ)
                    disable_tqdm = False):
    #
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id != -1 else "cpu"
    y = torch.from_numpy(y[None, :]).to(device, non_blocking=True)
    B = torch.from_numpy(B).to(device, non_blocking=True)
    L = torch.from_numpy(L).to(device, non_blocking=True)
    if lams_opt_train is not None:
        betas_opt_train = torch.from_numpy(betas_opt_train).to(device, non_blocking = True)
        betas_opt_val = torch.from_numpy(betas_opt_val).to(device, non_blocking = True)
        LOSS_betas = torch.zeros(nepoch0, 2).to(device)
    n, J = B.size()
    dim_lam = 8
    model = Model(n+dim_lam, J, nhidden, depth, use_torchsort, sort_reg_strength).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr = eta0, amsgrad = amsgrad)
    opt2 = torch.optim.Adam(model.parameters(), lr = eta, amsgrad = amsgrad)
    #sch1 = torch.optim.lr_scheduler.StepLR(opt1, gamma = gamma, step_size = decay_step)
    # sch1 = torch.optim.lr_scheduler.ReduceLROnPlateau(opt1, 'min', factor = gamma, patience = patience, cooldown = cooldown, min_lr = 1e-7, threshold = 1e-5)
    # sch1 = torch.optim.lr_scheduler.CyclicLR(opt1, 1e-6, eta0, cycle_momentum=False, mode = "exp_range", gamma = gamma)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, gamma = gamma, step_size = decay_step)
    loss_fn = nn.functional.mse_loss
    # just realized that pytorch also did not do sort in batch
    LOSS = torch.zeros(nepoch, 4).to(device)
    LOSS0 = torch.zeros(nepoch0, 4).to(device)
    query_lams = [lam_lo, lam_up, (lam_lo + lam_up) / 2]
    train_loss = []
    early_stopping0 = EarlyStopping(patience = patience0, verbose = False, path = model_file)
    def aug(lam):
        return [lam, lam**(1/3), np.exp(lam), np.sqrt(lam), np.log(lam), 10*lam, lam**2, lam**3]
    for epoch in range(nepoch0):
        pbar0 = tqdm.trange(niter_per_epoch, desc="Training G(lambda)", disable=disable_tqdm)
        # if epoch < stage_nepoch0:
        #     lams = torch.ones((K, 1)).to(device) * lam_lo #* (lam_up + lam_lo) / 2
        # else:
        #     lams = torch.rand((K, 1)).to(device) * (lam_up - lam_lo) + lam_lo
        for ii in pbar0:
            lams = torch.rand((K, 1), device = device) * (lam_up - lam_lo) + lam_lo
            ys = torch.cat((y.repeat( (K, 1) ), lams, torch.pow(lams, 1/3), torch.exp(lams), torch.sqrt(lams), 
                                                torch.log(lams), 10*lams, torch.square(lams), torch.pow(lams, 3)), dim = 1) # repeat works regardless of y has been augmented via `y[None, :]`
            betas = model(ys)
            ypred = torch.matmul(betas, B.t())
            loss1_fit = loss_fn(ypred, y.repeat((K, 1)))
            # loss1_fit = loss_fn(ypred, y) # throw a warning without repeat, but it should be fine
            #                               Kx1               K x J
            loss1 = loss1_fit + torch.mean(lams * torch.square(torch.matmul(betas, L))) * J / n
            opt1.zero_grad()
            loss1.backward()
            opt1.step()
            train_loss.append(loss1.item())
            pbar0.set_postfix(iter = ii, loss = loss1.item())
            if ii == niter_per_epoch - 1:
                LOSS0[epoch, 0] = loss1.item()
        
        for i in range(3):
            lam = query_lams[i]
            aug_lam = torch.tensor(aug(lam), dtype=torch.float32, device = device)
            ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
            beta = model(ylam)
            ypred = torch.matmul(beta, B.t())
            LOSS0[epoch, i+1] = loss_fn(ypred, y) + lam * torch.square(torch.matmul(beta, L)).mean() * J / n
        print(f"epoch = {epoch}, L(lam) = {LOSS0[epoch, 0]:.6f}, L(lam_lo) = {LOSS0[epoch, 1]:.6f}, L(lam_up) = {LOSS0[epoch, 2]:.6f}")
        if lams_opt_train is not None:
            loss_betas = []
            for i, lam in enumerate(lams_opt_train):
                aug_lam = torch.tensor(aug(lam), dtype=torch.float32, device = device)
                ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
                beta = model(ylam)
                loss_betas.append(loss_fn(betas_opt_train[i, :], beta[0]).item()) # beta is 1xJ
            LOSS_betas[epoch, 0] = np.mean(loss_betas)

            loss_betas = []
            for i, lam in enumerate(lams_opt_val):
                aug_lam = torch.tensor(aug(lam), dtype=torch.float32, device = device)
                ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
                beta = model(ylam)
                loss_betas.append(loss_fn(betas_opt_val[i, :], beta[0]).item())
            LOSS_betas[epoch, 1] = np.mean(loss_betas)

        # sch1.step()
        if not disable_early_stopping:
            early_stopping0(LOSS0[epoch, 1:].mean(), model)
            if early_stopping0.early_stop:
                print("Early stopping!")
                LOSS0 = LOSS0[:epoch,:]
                break
    # ##########
    # step 2 
    # ##########
    if not disable_early_stopping:
        model.load_state_dict(torch.load(model_file))
    model0 = copy.deepcopy(model)
    early_stopping = EarlyStopping(patience = patience, verbose = False, path = model_file)
    # for epoch in range(nepoch):
    for epoch in range(nepoch):
        pbar = tqdm.trange(niter_per_epoch, desc="Training G(y, lambda)", disable=disable_tqdm)
        for ii in pbar:
            if step2_use_tensor:
                # construct tensor
                lam = torch.rand((K0, 1, 1)) * (lam_up - lam_lo) + lam_lo
                aug_lam = torch.cat(aug(lam), dim=2).to(device, non_blocking=True) # K0 x 1 x 8
                ylam = torch.cat((y.repeat(K0, 1, 1), aug_lam), dim=2) # K0 x 1 x (n+8)
                # K0 x 1 x J
                if eval_sigma_adaptive:
                    beta = model(ylam).detach()
                else:
                    beta = model0(ylam).detach()
                # K0 x 1 x n
                ypred = torch.matmul(beta, B.t())
                sigma = torch.std(ypred - y, unbiased = True, dim = 2, keepdim = True) # K0 x 1 x 1 (keepdim), otherwise K0 x 1
                epsilons = torch.randn((K0, K, n), device = device) * sigma

                # construct training dataset
                ytrain = ypred + epsilons # K0 x K x n
                yslam = torch.cat((ytrain, aug_lam.repeat((1, K, 1))), dim=2) # K0 x K x (n+8)
                betas = model(yslam) # K0 x K x J
                yspred = torch.matmul(betas, B.t()) # K0 x K x n
                #                               K0x1x1                           K0xKxJ JxJ
                loss2 = loss_fn(yspred, ytrain) + torch.mean(lam.to(device) * torch.square(torch.matmul(betas, L))) * J / n
            else: # deprecated
                loss2 = 0
                for i in range(K0): # for each lam
                    lam = np.random.rand() * (lam_up - lam_lo) + lam_lo
                    aug_lam = torch.tensor(aug(lam), dtype=torch.float32).to(device)
                    ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
                    if eval_sigma_adaptive:
                        beta = model(ylam) 
                    else:
                        beta = model0(ylam) # do not influence by step 2
                    ypred = torch.matmul(beta, B.t())
                    sigma = torch.std(ypred.detach() - y, unbiased = True)
                    epsilons = torch.randn((K, n)).to(device) * sigma

                    #        1xn      +      Kxn
                    ytrain = ypred.detach() + epsilons 
                    yslam = torch.cat((ytrain, aug_lam.repeat((K, 1)) ), dim=1)
                    betas = model(yslam) # K x J
                    yspred = torch.matmul(betas, B.t()) # KxJ x Jxn
                    # ...............................................................KxJ x JxJ
                    loss2 = loss2 + loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() * J / n
                #
                loss2 = loss2 / K0
            opt2.zero_grad()
            loss2.backward()
            opt2.step()
            # sch2.step()
            train_loss.append(loss2.item())
            pbar.set_postfix(iter = ii, loss = loss2.item())
            if ii == niter_per_epoch - 1:
                LOSS[epoch, 0] = loss2.item()
        for i in range(3):
            lam = query_lams[i]
            aug_lam = torch.tensor(aug(lam), dtype=torch.float32, device = device)
            ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
            beta = model(ylam).detach()
            ypred = torch.matmul(beta, B.t())
            LOSS[epoch, i+1] = loss_fn(ypred, y) + lam * torch.square(torch.matmul(beta, L)).mean() * J / n

        sch2.step()
        print(f"epoch = {epoch}, L(lam) = {LOSS[epoch, 0]:.6f}, L(lam_lo) = {LOSS[epoch, 1]:.6f}, L(lam_up) = {LOSS[epoch, 2]:.6f}, lr = {sch2.get_last_lr()}")
        if not disable_early_stopping:
            early_stopping(LOSS[epoch, 1:].mean(), model)
            if early_stopping.early_stop:
                print("Early stopping!")
                LOSS = LOSS[:epoch,:]
                break
    #
    # pickle.dump([beta.cpu().detach().numpy(), ypred.cpu().detach().numpy()], open("debug-boot-step2.pl", "wb"))
    
    if disable_early_stopping:
        torch.save(model.state_dict(), model_file) # already saved as checkpoints in EarlyStopping (but it would problematic if larger than cooldown2)
    else:
        model.load_state_dict(torch.load(model_file))
    G = lambda y: model(torch.from_numpy(y[None,:]).to(device)).cpu().detach().numpy().squeeze() # y should be Float32
    loss_warmup = LOSS0.cpu().detach().numpy()
    loss_boot = LOSS.cpu().detach().numpy()
    if nepoch == 0:
        ret_loss = loss_warmup
    else:
        ret_loss = np.r_[loss_warmup, loss_boot]
    if lams_opt_train is not None:
        beta_loss = LOSS_betas.cpu().detach().numpy()
        #return G, train_loss, ret_loss, beta_loss
        return G, beta_loss, ret_loss # keep the same number of return parameters, then in Julia, LOSS is the beta_loss
    return G, train_loss, ret_loss

def load_model(n, dim_lam, J, nhidden, model_file, gpu_id = 3):
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    model = Model(n + dim_lam, J, nhidden).to(device)
    # https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html
    if device == "cpu":
        model.load_state_dict(torch.load(model_file, map_location=torch.device("cpu")))
    else:
        model.load_state_dict(torch.load(model_file))
    G = lambda y: model(torch.from_numpy(y[None,:]).to(device)).cpu().detach().numpy().squeeze()
    return G
