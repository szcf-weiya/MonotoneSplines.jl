using MonotoneSplines
using Plots

# First of all, generate data $y = exp(x) + ϵ$,
n = 20
σ = 0.1
x, y, x0, y0 = gen_data(n, σ, exp, seed = 1234);

# ## single lambda
# Here we train a MLP network $G(\lambda = λ_0)$ to approximate the solution $\hat\gamma_{\lambda_0}$.
λ = 1e-5
@time Ghat, loss = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, prop_nknots = 0.2);

# plot the training loss
plot(loss)

# Evaluate at $λ$,
yhat = Ghat(y, λ);

# compare it with the optimization solution
βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = 0.2);

# plot the fitted curves
scatter(x, y, label = "")
plot!(x0, y0, label = "truth", legend = :topleft, ls = :dot)
plot!(x, yhat, label = "MLP generator", ls = :dash, lw = 2)
plot!(x, yhat0, label = "OPT solution")
# The fitting curves obtained from optimization solution and MLP generator overlap quite well.

# ## lambda interval
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
