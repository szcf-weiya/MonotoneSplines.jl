using MonotoneSplines

using Plots

# First of all, generate data $y = exp(x) + ϵ$,
n = 20
σ = 0.1
x, y, x0, y0 = gen_data(n, σ, exp, seed = 1234);

# ## single lambda
# Here we train a MLP network $G(\lambda = λ_0)$ to approximate the solution $\hat\gamma_{\lambda_0}$.
λ = 1e-5
λl = 1e-2
λu = 1e-1
@time Ghat1, loss1 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1e-4);

@time Ghat2, loss2 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1e-1);

@time Ghat3, loss3 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=true, sort_reg_strength=1.0);

@time Ghat4, loss4 = mono_ss_mlp(x, y, λl = λl, λu = λu, device = :cpu, prop_nknots = 0.2, backend = "pytorch", 
                                    use_torchsort=false, sort_reg_strength=1.0);


λ = λl
yhat1 = Ghat1(y, λ)
yhat2 = Ghat2(y, λ)
yhat3 = Ghat3(y, λ)
yhat4 = Ghat4(y, λ)

scatter(x, y)
plot!(x, yhat1, label = "1e-4")
plot!(x, yhat2, label = "1e-1")
plot!(x, yhat3, label = "1")
plot!(x, yhat4, label = "no")

plot(loss1[1:100], label = "1e-4")
plot!(loss2[1:100], label = "1e-1")
plot!(loss3[1:100], label = "1")
plot!(loss4[1:100], label = "no")