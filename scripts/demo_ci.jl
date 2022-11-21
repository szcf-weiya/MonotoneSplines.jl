using Plots
using Random
using MonotoneSplines
n = 100
σ = 0.2
seed = 175
f = exp
x = rand(MersenneTwister(seed), n) * 2 .- 1
y = f.(x) + randn(MersenneTwister(seed), n) * σ
λ = 0.00063
λ = 0.13
yhat, YCI = MonotoneSplines.ci_mono_ss(x, y, λ, prop_nknots=0.2)

idx = sortperm(x)
cp = MonotoneSplines.coverage_prob(YCI, f.(x))
scatter(x, y)
plot!(x[idx], yhat[idx], label="λ = $λ")
plot!(x[idx], YCI[idx, 1], label="lower (cov prob = $cp)", ls = :dash)
plot!(x[idx], YCI[idx, 2], label="upper", ls = :dash)