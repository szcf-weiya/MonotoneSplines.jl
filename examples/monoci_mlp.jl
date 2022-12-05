using MonotoneSplines

n = 20
σ = 0.1
x, y, x0, y0 = gen_data(n, σ, exp, seed = 1234);

λl = 1e-2
λu = 1e-1
λs = range(λl, λu, length = 2)

@time RES0 = [ci_mono_ss(x, y, λ, prop_nknots = 0.2) for λ in λs]
Yhat0 = hcat([RES0[i][1] for i=1:2]...)
YCIs0 = [RES0[i][2] for i = 1:2]

@time Yhat, YCIs, LOSS = ci_mono_ss_mlp(x, y, λs, prop_nknots = 0.2, device = :cpu, backend = "flux", nepoch0 = 5, nepoch = 5);

@time Yhat2, YCIs2, LOSS2 = ci_mono_ss_mlp(x, y, λs, prop_nknots = 0.2, device = :cpu, backend = "pytorch", nepoch0 = 5, nepoch = 5);

# plot the traceplot of training loss
plot(log.(LOSS), label = "MLP generator (Flux)")
plot!(log.(LOSS2), label = "MLP generator (PyTorch)")

# Calculate the jaccard index
# OPT solution vs MLP generator (Flux)
[MonotoneSplines.jaccard_index(YCIs[i], YCIs0[i]) for i = 1:2]

# OPT solution vs MLP generator (PyTorch)
[MonotoneSplines.jaccard_index(YCIs2[i], YCIs0[i]) for i = 1:2]


scatter(x, y, label = "")
plot!(x0, y0, label = "truth", legend = :topleft, ls = :dot)
plot!(x, Yhat0[:, 1], label = "OPT solution")
plot!(x, Yhat0[:, 2], label = "OPT solution")
plot!(x, YCIs0[1][:, 1], fillrange = YCIs0[1][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, YCIs0[2][:, 1], fillrange = YCIs0[2][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, Yhat[:, 1], label = "MLP generator (Flux)", ls = :dash)
plot!(x, Yhat[:, 2], label = "MLP generator (Flux)", ls = :dash)
plot!(x, YCIs[1][:, 1], fillrange = YCIs[1][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, YCIs[2][:, 1], fillrange = YCIs[2][:, 2], linealpha = 0, label = "", fillalpha = 0.5)


scatter(x, y, label = "")
plot!(x0, y0, label = "truth", legend = :topleft, ls = :dot)
plot!(x, Yhat0[:, 1], label = "OPT solution")
plot!(x, Yhat0[:, 2], label = "OPT solution")
plot!(x, YCIs0[1][:, 1], fillrange = YCIs0[1][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, YCIs0[2][:, 1], fillrange = YCIs0[2][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, Yhat2[:, 1], label = "MLP generator (PyTorch)", ls = :dash)
plot!(x, Yhat2[:, 2], label = "MLP generator (PyTorch)", ls = :dash)
plot!(x, YCIs2[1][:, 1], fillrange = YCIs2[1][:, 2], linealpha = 0, label = "", fillalpha = 0.5)
plot!(x, YCIs2[2][:, 1], fillrange = YCIs2[2][:, 2], linealpha = 0, label = "", fillalpha = 0.5)

