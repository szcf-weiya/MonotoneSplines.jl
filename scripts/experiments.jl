
cubic(x) = x^3
logit(x) = 1/(1+exp(-x)) # TODO: export the functions from packages
logit5(x) = 1/(1+exp(-5x))
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

# cv true
for σ in [0.01, 0.1, 0.2, 0.5, 1.0]
    for f in [logit, logit5, exp, cubic]
        check_CI(n=100, σ = σ, f = f, nrep = 20, η = 1e-5, nepoch=100000, K = 200, λ = 0.0, nB=10000, fig=false, cvλ = true, λs = exp.(range(-10, 0, length = 100)), γ = 1.0, η0 = 1e-7, max_norm=1, clip_ratio=1)
    end
end



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
