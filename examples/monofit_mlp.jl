# This section illustrates how to use the MLP generator to perform the monotone fitting. The MLP generator can achieve a perfect approximation to the fitting curve obtained from the optimization toolbox quickly. Particulaly, the MLP generator can save time by avoiding repeating to run the optimization toolbox for continuous $\lambda$ since it only needs to train once to obtain the function $G(\lambda)$, which can immediately return the solution at $\lambda=\lambda_0$ by simply evaluating $G(\lambda_0)$.
using MonotoneSplines
using Plots

# We want to train a MLP generator $G(λ)$ to approximate the solution for the monotone spline.
# ```math
# \def\bfy{\mathbf{y}}
# \def\bB{\mathbf{B}}
# \def\bOmega{\boldsymbol{\Omega}}
# \def\subto{\mathrm{s.t.}}
# \begin{aligned}
# \min_{\gamma} & (\bfy - \bB\gamma)^T(\bfy - \bB\gamma) + \lambda\gamma^T\bOmega\gamma\\
# \subto\, & \alpha\gamma_1 \le \cdots \le \alpha\gamma_J 
# \end{aligned}
# ```

# First of all, generate data $y = \exp(x) + ϵ$,
n = 20
σ = 0.1
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, exp, seed = 1234);

# ## single $λ$
# Here we train a MLP network $G(\lambda = λ_0)$ to approximate the solution.
λ = 1e-5;

# By default, we use [Flux.jl](https://fluxml.ai/Flux.jl/stable/) deep learning framework,
Ghat, loss = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, disable_progressbar = true); # hide 
@time Ghat, loss = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, disable_progressbar = true);

# we also support the well-known PyTorch backend with the help of `PyCall.jl`,
@time Ghat2, loss2 = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, backend = "pytorch", disable_progressbar = true);

# !!! note
#     Showing the progressbar is quite useful in practice, but here in the documenter environment, it cannot display properly, so currently I simply disable it via `disable_progressbar = true`.

# plot the log training loss
plot(log.(loss), label = "Flux")
plot!(log.(loss2), label = "Pytorch")

# The fitting can be obtained via evaluating at $λ$,
yhat = Ghat(y, λ);
yhat2 = Ghat2(y, λ);

# compare it with the optimization solution
βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = 0.2);

# plot the fitted curves
scatter(x, y, label = "")
plot!(x0, y0, label = "truth", legend = :topleft, ls = :dot)
plot!(x, yhat, label = "MLP generator (Flux)", ls = :dash, lw = 2)
plot!(x, yhat0, label = "OPT solution")
plot!(x, yhat2, label = "MLP generator (Pytorch)", ls = :dash, lw = 2)
# The fitting curves obtained from optimization solution and MLP generator overlap quite well.

# ## continus $λ$
# Here we train a generator $G(\lambda), \lambda\in [\lambda_l, \lambda_u]$,
λl = 1e-2
λu = 1e-1
@time Ghat, loss = mono_ss_mlp(x, y, λl = λl, λu = λu, prop_nknots = 0.2, device = :cpu);

# Plot the training losses along with the iterations.
plot(loss)

# Evaluate the generator at $\lambda_l$, $\lambda_u$ and their middle $\lambda_m$
λm = (λl + λu) / 2
yhat_l = Ghat(y, λl)
yhat_u = Ghat(y, λu)
yhat_m = Ghat(y, λm)
_, yhat0_l = mono_ss(x, y, λl, prop_nknots = 0.2);
_, yhat0_u = mono_ss(x, y, λu, prop_nknots = 0.2);
_, yhat0_m = mono_ss(x, y, λm, prop_nknots = 0.2);

# Plot the fitting curves
scatter(x, y, label = "")
plot!(x0, y0, label = "truth", legend = :topleft, ls = :dot)
plot!(x, yhat0_l, label = "OPT solution (λ = $λl)")
plot!(x, yhat_l, label = "MLP generator (λ = $λl)", ls = :dash, lw = 2)
plot!(x, yhat0_m, label = "OPT solution (λ = $λm)")
plot!(x, yhat_m, label = "MLP generator (λ = $λm)", ls = :dash, lw = 2)
plot!(x, yhat0_u, label = "OPT solution (λ = $λu)")
plot!(x, yhat_u, label = "MLP generator (λ = $λu)", ls = :dash, lw = 2)
