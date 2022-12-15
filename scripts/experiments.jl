
cubic(x) = x^3
logit(x) = 1/(1+exp(-x)) # TODO: export the functions from packages
logit5(x) = 1/(1+exp(-5x))
sinhalfpi(x) = sin(pi/2 * x)
σs = [0.01, 0.1, 0.2, 0.5]
fs = [logit, exp, cubic]

check_CI(n=100, σ = 0.1, f = logit, nrep = 100, η = 0.001, nepoch=10000, K = 100, λ = 0.0, nB=10000)

check_CI(n=100, σ = 0.1, f = logit, nrep = 1, η = 0.001, nepoch=100, K = 100, λ = 0.0, nB=10000)

# gpu019-3 2022-10-18 23:43:27
for σ in [0.01, 0.1, 0.2, 0.5]
    for f in [logit, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 100, η = 0.001, nepoch=10000, K = 100, λ = 0.0, nB=10000, fig = false)
    end
end

# rocky 
for σ in [0.01, 0.1, 0.2, 0.5]
    for f in [logit, logit5, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 100, η = 0.001, η0 = 0.00001, nepoch=20000, K = 100, λ = 0.0, nB=10000, fig = false, γ = 1.0)
    end
end

# cv true 2022-10-21 23:41:50
for σ in [0.01, 0.1, 0.2, 0.5, 1.0]
    for f in [logit, logit5, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 20, η = 1e-5, nepoch=100000, K = 200, λ = 0.0, nB=10000, fig=false, cvλ = true, λs = exp.(range(-10, 0, length = 100)), γ = 1.0, η0 = 1e-7, max_norm=1, clip_ratio=1)
    end
end

# 2022-10-30 23:13:29
for σ in [0.01, 0.1, 0.2, 0.5, 1.0]
    for f in [logit, logit5, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 20, η = 1e-5, nepoch=100000, K = 100, λ = 0.0, nB=10000, fig=false, cvλ =false, figfolder=pwd(), λs = exp.(range(-10, 0, length = 100)), γ = 1.0, η0 = 1e-5, max_norm=1, clip_ratio=1, N1=1, N2=1)
    end
end

# gpu019 2022-11-01 18:10:11
for σ in [0.1, 0.2, 0.5, 1.0]
    for f in [logit, logit5, exp, cubic]
        check_acc(n=100, σ = σ, f = f, nrep = 20, η = 1e-3, M = 10, fig=true, figfolder=pwd(), λs = exp.(range(-10, -1, length = 10)), γ = 1.0, η0 = 1e-5, max_norm=1000, clip_ratio=1000, niter=300000)
    end
end

# gpu019 2022-11-01 18:28:43 (faster)
for σ in [0.1, 0.2, 0.5, 1.0]
    for f in [logit, logit5, exp, cubic]
        check_acc(n=100, σ = σ, f = f, nrep = 10, η = 1e-3, M = 10, fig=true, figfolder=pwd(), λs = exp.(range(-10, -1, length = 10)), γ = 1.0, η0 = 1e-5, max_norm=1000, clip_ratio=1000, niter=100000)
    end
end

# gpu019 2022-11-02 12:09:43
for σ in [0.1, 0.2, 0.5, 1.0]
    for f in [logit5, exp, cubic]
        η0 = 1e-4
        check_acc(n=100, σ = σ, f = f, nrep = 5, η = η0, M = 10, fig=true, figfolder=pwd(), λs = exp.(range(-10, -1, length = 10)), γ = 1.0, η0 = η0, max_norm=1000, clip_ratio=1000, niter=100000)
    end
end

# 2022-11-02 18:11:12
# 2022-11-02 23:12:14
# 2022-11-04 12:04:48
# 2022-11-04 15:55:12
# 2022-11-04 23:00:51
# 2022-11-07 16:39:52
# 2022-11-10 00:03:36
# 2022-11-12 12:52:13

for σ in [0.02, 0.1, 0.2, 0.5]
    for f in [logit5, exp, cubic, sinhalfpi]
        #η0 = 1e-4
        #check_acc(n=100, σ = σ, f = f, nrep = 5, η = η0, M = 100, fig=true, figfolder=pwd(), λs = exp.(range(-10, -1, length = 10)), γ = 1.0, η0 = η0, max_norm=1000, clip_ratio=1000, niter=500000)
        ## niter = 50000 each ~ 3.5min, then 3.5x5x3x3/60=2.6h
        ## niter = 100000 each ~ 7min, then 7x3x3x3/60=3.15h

        ## niter = 500000 each ~ 7x5 = 35min, then 35x3x3x3 = 15.75h
        ## niter = 300000 each ~ 7x3 = 21min, then 21x3x3x3 = 9.45h
        ## niter = 400000 each ~ 7x4 = 28min, then 28x10x3x3 = 42h
        ## niter = 1000000 each ~ 7x10 = 70min, then 70x10x2x2 = 46h (out of memory)
        ## niter = 800000 each ~ 7x8 = 56min, then 56x10x2x2 = 37h (out of memory)
        ## niter = 100000 each ~ 7min, then 1.5*7x5x3x2/60=5.25h
        ## niter = 150000 each ~ 10min, then 10x10x4x4/60=26h
        check_acc(n=100, σ = σ, f = f, nrep = 10, η = 1e-3, M = 100, fig=true, figfolder=pwd(), λs = exp.(range(-10, -1, length = 10)), γ = 0.5, η0 = 1e-4, max_norm=10, niter=150000, use_torchsort=false, gpu_id=3, patience=100, cooldown=2000000, nhidden=1000, depth=2, prop_nknots=0.2)
    end
end


# py"""
# import torch
# from boot import Model
# model2 = Model(108, 64, 1000)
# model2.load_state_dict(torch.load("model_G.pt"))
# G = lambda y: model2(torch.from_numpy(y[None,:])).cpu().detach().numpy().squeeze()
# """

# rocky 2022-10-19 12:45:38
for f in fs
    check_CI(nrep=10, nepoch=20000, λ=0.0, K=100, η = 0.001, f = f)
end

for f in fs
    for i = 1:10
        run(`convert fit-$f-$i.png loss-$f-$i.png +append $f-$i.png`)
    end
end
f = exp
for i = 1:5
    run(`convert fit-$f-$i.png loss-$f-$i.png +append $f-$i.png`)
end
run(`for i in {1..10}; do convert fit-$f-$i.png loss-$f-$i.png +append $f-$i.png; done`)

## 2022-12-08 12:45:51 (compare the running time)
for prop in [0.2]
    for n in [50, 100, 200, 500, 1000, 2000, 5000]
        check_CI(n=n, σ = 0.2, f = cubic, nrep = 2, η = 1e-4, η0 = 1e-4, check_acc=true, nepoch=100, nepoch0=5, K0=32, K =32, λ = 0.0, nB=2000, fig=true, cvλ =false, figfolder=pwd(), λs = exp.(range(-8, -2, length = 10)), γ = 0.9, method="lambda", prop_nknots=prop, seed=238, demo = true, gpu_id=2, niter_per_epoch=10000, nhidden=1000, decay_step=500, step2_use_tensor=true)
        # plot cannot be distinguished
        for i = 1:5
            run(`mv cubic-0.2-$i.png cubic-$n-$prop-$i.png`)
        end
    end
end

## 2022-12-11 14:27:35
prop=0.2
for σ in [0.1, 0.2, 0.5]
    for f in [logit5, exp, cubic, sinhalfpi]
        check_CI(n=100, σ = σ, f = f, nrep = 5, η = 1e-4, η0 = 1e-4, check_acc=true, nepoch=50, nepoch0=5, K0=32, K =32, λ = 0.0, nB=2000, fig=true, cvλ =false, figfolder=pwd(), λs = exp.(range(-8, -2, length = 10)), γ = 0.9, method="lambda", prop_nknots=prop, seed=238, demo = true, gpu_id=1, niter_per_epoch=10000, nhidden=1000, decay_step=500, step2_use_tensor=true)
    end
end

# ci 
prop = 0.2
for σ in [0.1, 0.2, 0.5]
    for f in [logit5, exp, cubic, sinhalfpi]
        check_CI(n=100, σ = σ, f = f, nrep = 5, η = 1e-4, η0 = 1e-4, check_acc=false, nepoch=50, nepoch0=5, K0=32, K =32, λ = 0.0, nB=2000, fig=true, cvλ =false, figfolder=pwd(), λs = exp.(range(-8, -2, length = 10)), γ = 0.9, method="lambda", prop_nknots=prop, seed=238, demo = true, gpu_id=7, niter_per_epoch=10000, nhidden=1000, decay_step=500, step2_use_tensor=true)
    end
end