# This section compares the beta loss when directly applying the classica MLP to train $(λ, β)$
using MonotoneSplines
using Plots
using Random
__init_pytorch__() # initialize supports for PyTorch backend

# Generate samples
seed = 1
n = 100
σ = 0.2
prop_nknots = 0.2
f = x -> x^3
x = rand(MersenneTwister(seed), n) * 2 .- 1
y = f.(x) + randn(MersenneTwister(seed+1), n) * σ
# construct B spline basis matrix
B, L, J = MonotoneSplines.build_model(x, prop_nknots = prop_nknots)

# Range of the smoothness parameter λ
λs = 10 .^ (range(-6, 0, length = 10))
λmin = minimum(λs)
λmax = maximum(λs)

# Set number of λ for training and validation
N = 100
ngrid = 100
arr_λs = rand(N) * (λmax - λmin) .+ λmin
βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in arr_λs]
βs_mat = vcat(βs'...)'
λs_mat = vcat(MonotoneSplines.aug.(arr_λs)'...)'

# for integrative loss
grid_λs = range(λmin, λmax, length=ngrid)
grid_βs = [mono_ss(x, y, λ, prop_nknots = prop_nknots).β for λ in grid_λs]
βs_grid_mat = vcat(grid_βs'...)'
λs_grid_mat = vcat(MonotoneSplines.aug.(grid_λs)'...)'

# train GpBS
G, Loss, Loss1, bfun = MonotoneSplines.py_train_G_lambda(y, B, L, nepoch = 0, nepoch0 = 3, K = N, 
            λl = λmin, λu = λmax, gpu_id = -1,
            nhidden = 1000, 
            λs_opt_train = λs, λs_opt_val = grid_λs, 
            βs_opt_train = βs_mat', βs_opt_val = βs_grid_mat', 
            niter_per_epoch = 200)

# Check training loss
plot(Loss, label = "training loss")

# Check beta loss
plot(Loss1[:, 1], label = "training L2 beta Loss")
plot!(Loss1[:, 2], label = "expected L2 beta loss")