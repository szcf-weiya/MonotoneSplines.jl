
σs = [0.01, 0.1, 0.2, 0.5]
fs = [logit, exp, cubic]
cubic(x) = x^3
logit(x) = 1/(1+exp(-x)) # TODO: export the functions from packages

check_CI(n=100, σ = 0.1, f = logit, nrep = 100, η = 0.001, nepoch=10000, K = 100, λ = 0.0, nB=10000)

check_CI(n=100, σ = 0.1, f = logit, nrep = 1, η = 0.001, nepoch=100, K = 100, λ = 0.0, nB=10000)

# gpu019-3 2022-10-18 23:43:27
for σ in [0.01, 0.1, 0.2, 0.5]
    for f in [logit, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 100, η = 0.001, nepoch=10000, K = 100, λ = 0.0, nB=10000, fig = false)
    end
end