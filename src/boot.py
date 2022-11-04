import torch
import torch.nn as nn
import numpy as np
import tqdm as tqdm
import pickle
import torchsort
# https://github.com/pytorch/pytorch/issues/45038#issuecomment-695793213
# too slow
# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'

class Model(nn.Module):
    def __init__(self, n, J, nhidden, use_torchsort = False, sort_reg_strength = 0.1):
        super(Model, self).__init__()
        # if y is a vector, suppose it has been added 1 dim such that the batch size = 1.
        # n = len(y[0])
        # n, J = B.size()
        self.use_torchsort = use_torchsort
        self.sort_reg_strength = sort_reg_strength
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
        if self.use_torchsort:
            beta = torchsort.soft_sort(beta_unsort, regularization_strength=self.sort_reg_strength)
        else:
            beta, indices = torch.sort(beta_unsort)
        return beta


# two different optimizers
# support lambda
def train_G(y, B, L, lam, K = 10, nepoch = 100, nhidden = 1000, eta = 0.001, eta0 = 0.0001, gamma = 0.9, sigma = 1, amsgrad = False, decay_step = 1000, max_norm = 2.0, clip_ratio = 1.0, debug_with_y0 = False, y0 = 0, warm_up = 100, subsequent_steps = True, N1 = 100, N2 = 100, gpu_id = 0):
    #
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        # torch.cuda.set_device(3)
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    y = torch.from_numpy(y[None, :]).to(device)
    if debug_with_y0:
        y0 = torch.from_numpy(y0[None, :]).to(device)
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
    n12 = max(N1, N2)
    LOSS = torch.zeros(nepoch*n12, 2)
    LOSS0 = torch.zeros(warm_up)
    pbar0 = tqdm.trange(warm_up, desc="Training G (warm up)")
    for epoch in pbar0:
        beta = model(y)
        ypred = torch.matmul(beta, B.t())
        loss1_fit = loss_fn(ypred, y)
        loss1 = loss1_fit + lam * torch.square(torch.matmul(beta, L)).mean() * J
        opt1.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        opt1.step()
        LOSS0[epoch] = loss1.item()
        pbar0.set_postfix(epoch = epoch, loss = loss1.item())

    pbar = tqdm.trange(nepoch, desc="Training G")
    # for epoch in range(nepoch):
    for epoch in pbar:
        if N1 < 100:
            pbar1 = range(N1)
        else:
            pbar1 = tqdm.trange(N1, desc="Training G (step 1)")
        for i in pbar1:
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
            LOSS[epoch*n12+i, 0] = loss1.item()
            if N1 >= 100:
                pbar1.set_postfix(i = i, loss = loss1.item(), sigma = sigma)
        if N2 < 100:
            pbar2 = range(N2)
        else:
            pbar2 = tqdm.trange(N2, desc="Training G (step 2)")
        beta = model(y)
        # ypred = torch.matmul(beta, B.t()) # update for next iteration in step 2
        # ypred = torch.matmul(beta, B.t()).detach() # update for next iteration in step 2
        ypred = torch.matmul(beta, B.t()).clone().detach() # update for next iteration in step 2
        sigma = torch.std(ypred - y, unbiased = True)
        for i in pbar2:
            # beta = model(y)
            # ypred = torch.matmul(beta, B.t()) # update for next iteration in step 2
            # sigma = torch.std(ypred.detach() - y, unbiased = True)
            epsilons = torch.randn((K, n)).to(device) * sigma
            #        1xn      +      Kxn
            # https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics
            # ytrain = ypred.detach() + epsilons  ## TODO: did ypred would be updated even if it put outside?
            ytrain = ypred + epsilons
            betas = model(ytrain) # K x J
            yspred = torch.matmul(betas, B.t()) # KxJ x Jxn
            # ...............................................................KxJ x JxJ
            loss2 = loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() * J
            #
            opt2.zero_grad()
            loss2.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm * clip_ratio)
            # nn.utils.clip_grad_value_(model.parameters(), max_norm)
            opt2.step()
            sch2.step()
            LOSS[epoch*n12+i, 1] = loss2.item() 
            # hete sigmas
#            sigma = torch.std(yspred.detach() - y, axis = 0, unbiased = True)
            # sigma = torch.mean(torch.abs(yspred.detach() - y), axis = 0) * np.sqrt(np.pi / 2)
            # pbar2.set_postfix(i = i, loss = loss2.item(), sigmas = [torch.min(sigma), torch.mean(sigma), torch.max(sigma)])
            run_mean = torch.mean(LOSS[epoch*n12:epoch*n12+i+1, 1]) # account for the sudden change after step 1
            if N2 >= 100:
                pbar2.set_postfix(i = i, loss = loss2.item(), running_mean = run_mean, sigma = sigma)
            #
        # LOSS[epoch, 0] = loss1.item()
        # LOSS[epoch, 1] = loss2.item()
        # print(f"epoch = {epoch}, loss = {LOSS[epoch,]}, sigma = {sigma}")
        # pickle.dump([yspred.cpu().detach().numpy(), ypred.cpu().detach().numpy(), y.cpu().detach().numpy()], open("debug-boot.pl", "wb"))
        pbar.set_postfix(epoch = epoch, loss = [LOSS[epoch*n12+N1-1, 0], LOSS[epoch*n12+N2-1, 1]], run_mean = run_mean, lr1 = sch1.get_last_lr())
        #if LOSS[epoch*n12+N2-1, 1] < LOSS[epoch*n12+N1-1, 0]:
        if max(LOSS[epoch*n12+N2-1, 1], run_mean) < LOSS[epoch*n12+N1-1, 0] and LOSS[epoch*n12+N2-1, 1] < run_mean:
            break
    #
    #
    # pickle.dump([beta.cpu().detach().numpy(), ypred.cpu().detach().numpy()], open("debug-boot-step2.pl", "wb"))
    G = lambda y: model(torch.from_numpy(y).to(device)).cpu().detach().numpy() # support y is Float32
    loss_warmup = LOSS0.cpu().detach().numpy()
    loss_boot = LOSS.cpu().detach().numpy()
    return G, np.r_[np.c_[loss_warmup, loss_warmup], loss_boot]

def train_G_lambda(y, B, L, K = 10, nepoch = 100, nhidden = 1000, eta = 0.001, eta0 = 0.0001, gamma = 0.9, sigma = 1, amsgrad = False, decay_step = 1000, max_norm = 2.0, clip_ratio = 1.0, debug_with_y0 = False, y0 = 0, warm_up = 100, subsequent_steps = True, N1 = 100, N2 = 100, lam_lo = 1e-9, lam_up = 1e-4, use_torchsort = False, sort_reg_strength = 0.1, gpu_id = 0):
    #
    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(y[None, :]).to(device)
    if debug_with_y0:
        y0 = torch.from_numpy(y0[None, :]).to(device)
    B = torch.from_numpy(B).to(device)
    L = torch.from_numpy(L).to(device)
    n, J = B.size()
    dim_lam = 8
    model = Model(n+dim_lam, J, nhidden, use_torchsort, sort_reg_strength).to(device)
    opt1 = torch.optim.Adam(model.parameters(), lr = eta0, amsgrad = amsgrad)
    opt2 = torch.optim.Adam(model.parameters(), lr = eta, amsgrad = amsgrad)
    sch1 = torch.optim.lr_scheduler.StepLR(opt1, gamma = gamma, step_size = decay_step)
    sch2 = torch.optim.lr_scheduler.StepLR(opt2, gamma = gamma, step_size = decay_step)
    loss_fn = nn.functional.mse_loss
    # just realized that pytorch also did not do sort in batch
    LOSS = torch.zeros(nepoch).to(device)
    LOSS0 = torch.zeros(warm_up, 4).to(device)
    query_lams = [lam_lo, lam_up, (lam_lo + lam_up) / 2]
    pbar0 = tqdm.trange(warm_up, desc="Training G (warm up)")
    for epoch in pbar0:
        lams = torch.rand((K, 1)).to(device) * (lam_up - lam_lo) + lam_lo
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
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        opt1.step()
        LOSS0[epoch, 0] = loss1.item()
        pbar0.set_postfix(epoch = epoch, loss = loss1.item())
        for i in range(3):
            lam = query_lams[i]
            aug_lam = torch.tensor([lam, lam**(1/3), np.exp(lam), np.sqrt(lam), np.log(lam), 10*lam, lam**2, lam**3], dtype=torch.float32).to(device)
            ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
            beta = model(ylam)
            ypred = torch.matmul(beta, B.t())
            LOSS0[epoch, i+1] = loss_fn(ypred, y) + lam * torch.square(torch.matmul(beta, L)).mean() * J / n
    # G = lambda y: model(torch.from_numpy(y).to(device)).cpu().detach().numpy() # support y is Float32

    # return G, LOSS0.cpu().detach().numpy()
    pbar = tqdm.trange(nepoch, desc="Training G")
    # for epoch in range(nepoch):
    for epoch in pbar:
        loss2 = 0
        for i in range(K): # for each lam
            lam = np.random.rand() * (lam_up - lam_lo) + lam_lo
            aug_lam = torch.tensor([lam, lam**(1/3), np.exp(lam), np.sqrt(lam), np.log(lam), 10*lam, lam**2, lam**3], dtype=torch.float32).to(device)
            ylam = torch.cat((y, aug_lam.repeat((1, 1))), dim=1)
            beta = model(ylam)
            ypred = torch.matmul(beta, B.t())
            sigma = torch.std(ypred.detach() - y, unbiased = True)
            epsilons = torch.randn((K, n)).to(device) * sigma

            #        1xn      +      Kxn
            # https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics
            # ytrain = ypred.detach() + epsilons  ## TODO: did ypred would be updated even if it put outside?
            ytrain = ypred + epsilons
            yslam = torch.cat((ytrain, aug_lam.repeat((K, 1)) ), dim=1)
            betas = model(yslam) # K x J
            yspred = torch.matmul(betas, B.t()) # KxJ x Jxn
            # ...............................................................KxJ x JxJ
            loss2 = loss2 + loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() * J / n
        #
        opt2.zero_grad()
        loss2.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm * clip_ratio)
        # nn.utils.clip_grad_value_(model.parameters(), max_norm)
        opt2.step()
        sch2.step()
        LOSS[epoch] = loss2.item() / K
        pbar.set_postfix(epoch = epoch, loss = LOSS[epoch])
    #
    # pickle.dump([beta.cpu().detach().numpy(), ypred.cpu().detach().numpy()], open("debug-boot-step2.pl", "wb"))
    # https://stackoverflow.com/a/43819235/
    # https://github.com/pytorch/pytorch/blob/761d6799beb3afa03657a71776412a2171ee7533/docs/source/notes/serialization.rst
    # torch.save(model.state_dict(), "model_G.pt") 
    # model2 = Model(n+dim_lam, J, nhidden, use_torchsort, sort_reg_strength)
    # model2.load_state_dict(torch.load("model_G.pt"))
    G = lambda y: model(torch.from_numpy(y[None,:]).to(device)).cpu().detach().numpy().squeeze() # support y is Float32
    # G = lambda y: model2(torch.from_numpy(y[None,:])).cpu().detach().numpy().squeeze() # support y is Float32
    loss_warmup = LOSS0.cpu().detach().numpy()
    loss_boot = LOSS.cpu().detach().numpy()
    if nepoch == 0:
        #ret_loss = loss_warmup[::10,:] # attempts to avoid StackOverflowError #120
        ret_loss = loss_warmup
    else:
        ret_loss = np.r_[loss_warmup, loss_boot]
    return G, ret_loss 

def train_G_bp(y, B, L, lam, K = 10, nepoch = 100, nhidden = 1000, eta = 0.001, eta0 = 0.0001, gamma = 0.9, sigma = 1, amsgrad = False, decay_step = 1000, max_norm = 2.0, clip_ratio = 1.0, debug_with_y0 = False, y0 = 0, warm_up = 100, subsequent_steps = True):
    #
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y = torch.from_numpy(y[None, :]).to(device)
    if debug_with_y0:
        y0 = torch.from_numpy(y0[None, :]).to(device)
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
    LOSS0 = torch.zeros(warm_up)
    pbar = tqdm.trange(nepoch, desc="Training G")
    pbar0 = tqdm.trange(warm_up, desc="Training G (warm up)")
    for epoch in pbar0:
        beta = model(y)
        ypred = torch.matmul(beta, B.t())
        loss1_fit = loss_fn(ypred, y)
        loss1 = loss1_fit + lam * torch.square(torch.matmul(beta, L)).mean() * J
        opt1.zero_grad()
        loss1.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        opt1.step()
        LOSS0[epoch] = loss1.item()
        pbar0.set_postfix(epoch = epoch, loss = loss1.item())
    beta0 = model(y)
    ypred0 = torch.matmul(beta0, B.t())
    sigma0 = torch.std(ypred0.detach() - y, unbiased = True)
    pickle.dump([beta0.cpu().detach().numpy(), ypred0.cpu().detach().numpy(), y.cpu().detach().numpy(), L.cpu().detach().numpy()], open("debug-boot-step1.pl", "wb"))
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
        if not debug_with_y0 or not subsequent_steps:
            opt1.zero_grad()
            loss1.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            # nn.utils.clip_grad_value_(model.parameters(), max_norm)
            opt1.step()
            sch1.step()
            #
            sigma = torch.std(ypred.detach() - y, unbiased = True)
        ## second step
        if debug_with_y0:
            sigma = torch.std(y0 - y, unbiased = True)
        if subsequent_steps:
            sigma = torch.std(ypred.detach() - y, unbiased = True)
            # if sigma > sigma0:
            #     ypred = torch.copysign(ypred0, ypred0)
            #     sigma = sigma0
        epsilons = torch.randn((K, n)).to(device) * sigma
        #        1xn      +      Kxn
        # https://pytorch.org/docs/master/notes/broadcasting.html#broadcasting-semantics
        ytrain = ypred.detach() + epsilons 
        if debug_with_y0:
            ytrain = y0 + epsilons 
        betas = model(ytrain) # K x J
        yspred = torch.matmul(betas, B.t()) # KxJ x Jxn
        # ...............................................................KxJ x JxJ
        pickle.dump([epsilons.cpu().detach().numpy(), ytrain.cpu().detach().numpy(), yspred.cpu().detach().numpy(), betas.cpu().detach().numpy()], open(f"debug-boot-{epoch}.pl", "wb"))
        loss2 = loss_fn(yspred, ytrain) + lam * torch.square(torch.matmul(betas, L)).mean() * J
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
    pickle.dump([beta.cpu().detach().numpy(), ypred.cpu().detach().numpy()], open("debug-boot-step2.pl", "wb"))
    G = lambda y: model(torch.from_numpy(y).to(device)).cpu().detach().numpy() # support y is Float32
    loss_warmup = LOSS0.cpu().detach().numpy()
    loss_boot = LOSS.cpu().detach().numpy()
    return G, np.r_[np.c_[loss_warmup, loss_warmup], loss_boot]


