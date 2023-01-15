# When using PyTorch backend in MLP generator, there are two choices for the `sort` operation:
# 
# - the default `torch.sort` operation whose "gradient" is defined following the [instruction for non-differentiable functions](https://pytorch.org/docs/stable/notes/autograd.html#gradients-for-non-differentiable-functions) 
# - a differentiable sort operation [`torchsort.soft_sort`](https://github.com/teddykoker/torchsort).
#
# This section will compare these two operations and show that their difference are neglectable.
using MonotoneSplines
using Plots

# First of all, generate data $y = \exp(x) + ϵ$,
n = 20
σ = 0.1
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, exp, seed = 1234);

# Here we train a MLP network $G(\lambda = λ_0)$ to approximate the solution $\hat\gamma_{\lambda_0}$ for a single $\lambda$.
λl = 1e-2
λu = λl
@time Ghat1, loss1 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1e-4, disable_progressbar = true);

@time Ghat2, loss2 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1e-1, disable_progressbar = true);

@time Ghat3, loss3 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1.0, disable_progressbar = true);

@time Ghat4, loss4 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=false, sort_reg_strength=1.0, disable_progressbar = true);

# Evaluate the fitted curve,
λ = λl
yhat1 = Ghat1(y, λ)
yhat2 = Ghat2(y, λ)
yhat3 = Ghat3(y, λ)
yhat4 = Ghat4(y, λ);

# The fitted curves are
scatter(x, y, label = "")
plot!(x, yhat1, label = "1e-4")
plot!(x, yhat2, label = "1e-1")
plot!(x, yhat3, label = "1")
plot!(x, yhat4, label = "no")

# And the traing loss is
plot(loss1[1:100], label = "1e-4", xlab = "iter", ylab = "loss")
plot!(loss2[1:100], label = "1e-1")
plot!(loss3[1:100], label = "1")
plot!(loss4[1:100], label = "no")