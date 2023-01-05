# rename to GpBS? (c.f. GBS)
using Flux
using Plots
using PyCall
using Serialization
using Random
using BSON
using Zygote
using ProgressMeter

## determine functions formally (NB: Be better not to change the name)
cubic(x) = x^3
logit(x) = 1/(1+exp(-x))
logit5(x) = 1/(1+exp(-5x))
sinhalfpi(x) = sin(pi/2 * x)

"""
    check_CI(; <keyword arguments>)

Conduct repeated experiments to check the overlap of confidence bands (default, `check_acc = false`) or accuracy of fitting curves (`check_acc = true`) between MLP generator and OPT solution. 

## Arguments

- `n = 100`: sample size
- `σ = 0.1`: noise level
- `f::Function = exp`: the truth curve
- `seed = 1234`: random seed for the simulated data
- `check_acc = false`: check overlap of confidence bands (default: false) or accuracy of fitting curves (true)
- `nepoch0 = 5`: number of epoch in the first step to fit the curve
- `nepoch = 50`: number of epoch in the second step to obtain the confidence band
- `niter_per_epoch = 100`: number of iterations in each epoch
- `η0 = 1e-4`: learning rate in step 1
- `η = 1e-4`: learning rate in step 2 (NOTE: lr did not make much difference, unify these two)
- `K0 = 32`: Monte Carlo size for averaging `λ` in step 2
- `K = 32`: Monte Carlo size for averaging `λ` in step 1 and for averaging `y` in step 2. (NOTE: unify these two Monte Carlo size)
- `nB = 2000`: number of bootstrap replications
- `nrep = 5`: number of repeated experiments
- `fig = true`: whether to plot
- `figfolder = ~`: folder for saving the figures if `fig = true`
- `λs = exp.(range(-8, -2, length = 10))`: region of continuous `λ`
- `nhidden = 1000`: number of hidden layers
- `depth = 2`: depth of MLP
- `demo = false`: whether to save internal results for demo purpose
- `model_file = nothing`: if not nothing, load the model from the file.
- `gpu_id = 0`: specify the id of GPU, -1 for CPU.
- `prop_nknots = 0.2`: proportion of number of knots in B-spline basis. 
- `backend = "flux"`: train MLP generator with Flux or PyTorch
"""
function check_CI(; n = 100, σ = 0.1, f = exp, seed = 1234, 
                    nepoch0 = 5, nepoch = 50, niter_per_epoch = 100, 
                    η0 = 1e-4, η = 1e-4, 
                    nrep = 5, α = 0.05, C = 1, 
                    backend = "pytorch",
                    K0 = 10, K = 10,
                    nB = 2000, 
                    λs = exp.(range(-8, -2, length = 10)),
                    γ = 0.9, # deprecated
                    demo = false, # save internal results for demo purpose
                    decay_step = 5, # deprecated
                    prop_nknots = 1.0,
                    nhidden = 1000, depth = 2,
                    gpu_id = 3,
                    model_file = nothing,
                    cooldown2 = 10, # deprecated
                    patience = 100, cooldown = 100, # deprecated
                    sort_in_nn = true, # only flux
                    check_acc = false, # reduce to check_acc
                    fig = true, figfolder = "~", kw...
                    )
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    if check_acc
        nepoch = 0
    end
    nλ = length(λs)
    res_time = zeros(nrep, 5) # OPT fit (discrete lambda, 20 lambda) & MLP generator (continue lambda, infty) & MLP eval
                             # OPT ci (single lambda) & MLP generator (continue lambda)
    res_err = zeros(nrep, 3) 
    Err_boot = zeros(nrep, nλ, 3)
    res_covprob = zeros(nrep, nλ, 2) # OPT and MLP
    res_overlap = zeros(nrep, nλ)
    res_err = zeros(nrep, nλ, 3)
    Yhat0 = zeros(nλ, n)
    Yhat = zeros(nλ, n)
    ## julia's progress bar has been overrideen by tqdm's progress bar
    for i = 1:nrep
        if nrep == 1
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed+1), n) * σ # to avoid the same random seed
        else
            seed = Int(rand(UInt8))
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed+1), n) * σ
            # x = rand(n) * 2 .- 1
            # y = f.(x) + randn(n) * σ    
        end
        B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
        ## optimization solution for monotone splines
        res_time[i, 1] = @elapsed for (j, λ) in enumerate(λs)
            βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = prop_nknots)
            Yhat0[j, :] = yhat0
        end
        ## train MLP generator
        res_time[i, 2] = @elapsed begin
            if isnothing(model_file)
                if backend == "flux"
                    model_file = "model-$f-$σ-n$n-J$J-nhidden$nhidden-$i-$seed-$timestamp.bson"
                    λl = λs[1]
                    λu = λs[end]
                    device = ifelse(gpu_id == -1, :cpu, :gpu)
                    Ghat0, loss0 = train_Gλ(y, B, L; λl = λl, λu = λu,
                                                    device = device, model_file = model_file, 
                                                    niter_per_epoch = niter_per_epoch,
                                                    nhidden = nhidden,
                                                    sort_in_nn = sort_in_nn,
                                                    M = K,
                                                    nepoch = nepoch0, kw...)
                    Ghat, loss = train_Gyλ(y, B, L, model_file; device = device, 
                                                    nepoch = nepoch, niter_per_epoch = niter_per_epoch,
                                                    M = K,
                                                    sort_in_nn = sort_in_nn,
                                                    λl = λl, λu = λu)
                    LOSS = vcat(loss0, loss)
                else # backend = "pytorch"
                    M = K
                    Ghat, LOSS = py_train_G_lambda(y,
                                                    B, L, K = M, nepoch = nepoch, η = η, 
                                                    K0 = K0,
                                                    γ = γ, η0 = η0, 
                                                    use_torchsort = false,
                                                    gpu_id = gpu_id,
                                                    nhidden = nhidden, depth = depth,
                                                    niter_per_epoch = niter_per_epoch, cooldown2 = cooldown2,
                                                    model_file = "model-$f-$σ-n$n-J$J-nhidden$nhidden-$i-$seed-$timestamp.pt",
                                                    patience = patience, cooldown = cooldown,
                                                    decay_step = decay_step, 
                                                    nepoch0 = nepoch0, λl = λs[1], λu = λs[end])
                end    
                if fig
                    savefig(plot(log.(LOSS)), "$figfolder/loss-$f-$σ-$i.png")
                end
            else
                # gpu is much faster
                Ghat = load_model(B, model_file, gpu_id = gpu_id)
            end
        end
        if fig
            fitfig = scatter(x, y, legend=:topleft, label="", title="seed = $seed, J = $J")
        end
        idx = sortperm(x)
        fit_err = zeros(nλ, n)
        RES_YCI0 = Array{Any, 1}(undef, nλ)
        for (j, λ) in enumerate(λs)
            res_time[i, 3] += @elapsed begin
                _, YCI = MonotoneSplines.ci_mono_ss(x, y, λ, prop_nknots=prop_nknots, B = nB)
            end
            RES_YCI0[j] = YCI
            res_time[i, 4] += @elapsed begin
                yhat = Ghat(y, λ)                
            end
            Yhat[j, :] .= yhat
            rel_gap = Flux.Losses.mse(Yhat0[j, :], yhat) / Flux.Losses.mse(Yhat0[j, :], zeros(n))
            fit_ratio = Flux.Losses.mse(yhat, y) / Flux.Losses.mse(Yhat0[j, :], y)
            fit_ratio2 = Flux.Losses.mse(yhat, f.(x)) / Flux.Losses.mse(Yhat0[j, :], f.(x))
            res_err[i, j, :] .= [rel_gap, fit_ratio, fit_ratio2]
            fit_err[j, :] .= (yhat - y).^2
            if fig
                plot!(fitfig, x[idx], yhat[idx], label = "λ = $λ")
                plot!(fitfig, x[idx], Yhat0[j, idx], label = "", ls = :dash)
            end
        end
        if fig
            savefig(fitfig, "$figfolder/fit-$f-$σ-$i.png")
        end
        res_time[i, 5] = @elapsed begin
            RES_YCI, cov_hat = sample_G_λ(Ghat, y, λs, nB = nB)               
        end
        cp = [coverage_prob(YCI, f.(x)) for YCI in RES_YCI]
        cp0 = [coverage_prob(YCI, f.(x)) for YCI in RES_YCI0]
        if demo
            serialize("demo-CI-$f-n$n-σ$σ-seed$seed-B$nB-K0$K0-K$K-nepoch$nepoch-prop$(prop_nknots)-$timestamp.sil", [x, y, λs, J, Yhat, Yhat0, RES_YCI, RES_YCI0, cp, cp0]) # loss not exist when from file
        end
        for j = 1:nλ
            res_overlap[i, j] = jaccard_index(RES_YCI[j], RES_YCI0[j])
        end
        res_covprob[i, :, 1] .= cp
        res_covprob[i, :, 2] .= cp0
        Err_boot[i, :, 1] .= mean(fit_err, dims=2)[:]
        Err_boot[i, :, 2] .= mean(cov_hat, dims=2)[:]
        Err_boot[i, :, 3] .= Err_boot[i, :, 1] + 2 * Err_boot[i, :, 2]
        errs, _, _, _ = cv_mono_ss(x, y, λs, nfold = 10)
        errs2, _, _, _ = cv_mono_ss(x, y, λs, nfold = n)
        if fig
            errfig = plot(log.(λs), Err_boot[i, :, 3], label = "err + 2cov")
            plot!(errfig, log.(λs), Err_boot[i, :, 1], label = "err")
            plot!(errfig, log.(λs), errs, label = "10 fold CV")
            plot!(errfig, log.(λs), errs2, label = "LOOCV")
            savefig(errfig, "$figfolder/err-$f-$σ-$i.png")
            savefig(plot(log.(λs), cp), "$figfolder/cp-$f-$σ-$i.png")    
        end
        if fig # TODO: cannot generalize
            if strip(read(`hostname`, String)) == "chpc-gpu019"
                run(`ssh sandbox convert $figfolder/cp-$f-$σ-$i.png $figfolder/err-$f-$σ-$i.png $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
            else
                run(`convert $figfolder/cp-$f-$σ-$i.png $figfolder/err-$f-$σ-$i.png $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
            end
        end                    
    end
    serialize("$f-n$n-σ$σ-nrep$nrep-B$nB-K$K-η$η-nepoch$nepoch-$timestamp.sil", [res_covprob, res_overlap, res_err, res_time, Err_boot])
    return mean(res_overlap, dims=1)[1,:], mean(res_covprob, dims=1), mean(res_err, dims=1)[1,:,:], mean(res_time)
end

"""
    aug(λ::AbstractFloat)

Augment `λ` with 8 different functions.
"""
aug(λ::AbstractFloat) = [λ, cbrt(λ), exp(λ), sqrt(λ), log(λ), 10*λ, λ^2, λ^3]

batch_sort(x::Matrix) = reduce(hcat, [sort(xi) for xi in eachcol(x)])

Zygote.@adjoint function batch_sort(x::Matrix)
    ps = []
    for xi in eachcol(x)
        p = sortperm(xi)
        push!(ps, p)
    end
    return reduce(hcat, [x[ps[i], i] for i in eachindex(ps)]), 
           x̄ -> (reduce(hcat, [x̄[invperm(ps[i]),i] for i in eachindex(ps)]),)
end

"""
    train_Gλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix; λl, λu)

Train MLP generator G(λ) for λ ∈ [λl, λu].
"""
function train_Gλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix; dim_λ = 8, 
                                        activation = gelu,
                                        nhidden = 100,
                                        M = 10,
                                        niter_per_epoch = 100, nepoch = 3,
                                        device = :cpu,
                                        model_file = "/tmp/model_G.bson",
                                        sort_in_nn = true,
                                        eval_in_batch = false, # TODO: need further checking (Clouds#143)
                                        disable_progressbar = false,
                                        λl = 1e-4, λu = 1e-3, kw...)
    device = eval(device)
    y = device(rawy)
    B = device(rawB)
    L = device(rawL)
    n, J = size(B)
    eval_in_batch = !sort_in_nn & eval_in_batch
    if sort_in_nn
        G = Chain(Dense(n + dim_λ => nhidden, activation), 
            Dense(nhidden => nhidden, activation),
            Dense(nhidden => nhidden, activation), 
            Dense(nhidden => J),
            sort
        ) |> device
    else
        G = Chain(Dense(n + dim_λ => nhidden, activation), 
            Dense(nhidden => nhidden, activation),
            Dense(nhidden => nhidden, activation), 
            Dense(nhidden => J)
        ) |> device
    end
    opt = AMSGrad()
    #loss(λ::AbstractFloat) = Flux.Losses.mse(B * G(vcat(y, device(aug(λ)) )), y) + λ * sum((L' * G(vcat(y, device(aug(λ)) )) ).^2) / n
    yaug(λ::AbstractFloat) = device(vcat(rawy, aug(λ)))
    loss(λ::AbstractFloat) = ifelse(sort_in_nn, Flux.Losses.mse(B * G(yaug(λ)) , y) + λ * sum((L' * G(yaug(λ)) ).^2) / n,
                            Flux.Losses.mse(rawB * sort(cpu(G(yaug(λ)))) , rawy) + λ * sum((rawL' * sort(cpu(G(yaug(λ)))) ).^2) / n)
    losses(λs::Vector) = mean([loss(λ) for λ in λs])
    # only support sort_in_nn = false (see #103)
    function loss_batch(λs::Vector)
        ys = repeat(rawy, 1, M)
        ys_aug = vcat(ys, reduce(hcat, aug.(λs))) # (n+8)*M
        Γ = batch_sort(cpu(G(device(ys_aug))))
        #                                               JxJ   JxM   Mx1
        return Flux.Losses.mse(rawB * Γ, ys) + sum((rawL' * Γ * sqrt.(λs)).^2) / n / M
    end
    train_loss = Float64[]
    for epoch in 1:nepoch
        p = Progress(niter_per_epoch, dt = 1, desc = "Train G(λ): ", enabled = !disable_progressbar)
        for i in 1:niter_per_epoch
            λs = rand(M) * (λu - λl) .+ λl
            if eval_in_batch
                Flux.train!(loss_batch, Flux.params(G), [(λs,)], opt)
                append!(train_loss, loss_batch(λs))
            else                    
                Flux.train!(losses, Flux.params(G), [(λs,)], opt)
                append!(train_loss, losses(λs))
            end
        end
    end
    G = cpu(G)
    BSON.@save model_file G
    if sort_in_nn
        return (y::AbstractVector{<:AbstractFloat}, λ::AbstractFloat) -> rawB * G(vcat(y, aug(λ))), train_loss
    else
        # no need batch_sort here
        return (y::AbstractVector{<:AbstractFloat}, λ::AbstractFloat) -> rawB * sort(G(vcat(y, aug(λ)))), train_loss
    end
end

"""
    train_Gyλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix, model_file::String)

Train MLP generator G(y, λ) for λ ∈ [λl, λu] and y ~ N(f, σ²)
"""
function train_Gyλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix, model_file::String; device = :cpu, 
                        niter_per_epoch = 100, nepoch = 3, λl = 1e-4, λu = 1e-3,
                        sort_in_nn = true,
                        disable_progressbar = false,
                        M = 10, kw...)
    device = eval(device)
    y = device(rawy)
    B = device(rawB)
    L = device(rawL)
    n, J = size(B)
    # G0 = BSON.@load model_file G (see issue #3)
    # G = BSON.@load model_file G
    G0 = BSON.load(model_file, @__MODULE__)[:G]
    G = BSON.load(model_file, @__MODULE__)[:G]
    G = device(G)
    opt = AMSGrad()
    train_loss = Float64[]
    function loss(λ::AbstractFloat)
        if sort_in_nn
            ypred = rawB * G0(vcat(rawy, aug(λ))) # G0 on cpu
        else
            ypred = rawB * sort(G0(vcat(rawy, aug(λ))))
        end
        σ = std(ypred - rawy)
        ytrain = [ypred + randn(n) * σ for _ in 1:M]
        if sort_in_nn
            return mean([Flux.Losses.mse(B * G(device(vcat(ytrain[i], aug(λ)) )), ytrain[i]) + λ * sum((L' * G(device(vcat(ytrain[i], aug(λ)) )) ).^2) / n for i = 1:M])
        else
            return mean([Flux.Losses.mse(rawB * sort(cpu(G(device(vcat(ytrain[i], aug(λ)))))), ytrain[i]) + λ * sum((rawL' * sort(cpu(G(device(vcat(ytrain[i], aug(λ)))))) ).^2) / n for i = 1:M])
        end
    end
    losses(λs::Vector) = mean([loss(λ) for λ in λs])
    for epoch in 1:nepoch
        p = Progress(niter_per_epoch, dt = 1, desc = "Train G(y, λ): ", enabled = !disable_progressbar)
        for i in 1:niter_per_epoch
            λs = rand(M) * (λu - λl) .+ λl
            Flux.train!(losses, Flux.params(G), [λs], opt)
            append!(train_loss, losses(λs))
            next!(p)
        end
    end
    G = cpu(G)
    model_file1 = model_file[1:end-5] * "_ci.bson" # keep G0
    BSON.@save model_file1 G
    if sort_in_nn
        return (y::AbstractVector{<:AbstractFloat}, λ::AbstractFloat) -> rawB * G(vcat(y, aug(λ))), train_loss
    else
        return (y::AbstractVector{<:AbstractFloat}, λ::AbstractFloat) -> rawB * sort(G(vcat(y, aug(λ)))), train_loss
    end
end



"""
    mono_ss_mlp(x::AbstractVector, y::AbstractVector; λl, λu)

Fit monotone smoothing spline by training a MLP generator.

## Arguments

- `prop_nknots = 0.2`: proportion of number of knots
- `backend = flux`: use `flux` or `pytorch`
- `device = :cpu`: use `:cpu` or `:gpu`
- `nhidden = 100`: number of hidden units
- `disable_progressbar = false`: disable progressbar (useful in Documenter.jl)
"""
function mono_ss_mlp(x::AbstractVector, y::AbstractVector; prop_nknots = 0.2, 
                                                        backend = "flux", λl = 1e-5, λu = 1e-4, 
                                                        device = :cpu, 
                                                        nhidden = 100,
                                                        disable_progressbar = false, # for pytorch backend (see #2)
                                                        kw...)
    B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
    if backend == "flux"
        Ghat, LOSS = train_Gλ(y, B, L; λl = λl, λu = λu, device = device, nhidden = nhidden, disable_progressbar = disable_progressbar, kw...)
    else
        Ghat, LOSS = py_train_G_lambda(y, B, L; nepoch = 0, nepoch0 = 3, 
                                                λl = λl, λu = λu, 
                                                disable_tqdm = disable_progressbar,
                                                nhidden = nhidden,
                                                gpu_id = ifelse(device == :cpu, -1, 0), kw...)
    end
    return Ghat, LOSS
end

"""
    ci_mono_ss_mlp(x::AbstractVector{T}, y::AbstractVector{T}, λs::AbstractVector{T}; )

Fit data `x, y` at each `λs` with confidence bands.

## Arguments

- `prop_nknots = 0.2`: proportion of number of knots
- `backend = "flux"`: `flux` or `pytorch`
- `model_file`: path for saving trained model
- `nepoch0 = 3`: number of epoch in training step 1
- `nepoch = 3`: number of epoch in training step 2
- `niter_per_epoch = 100`: number of iterations in each epoch
- `M = 10`: Monte Carlo size
- `nhidden = 100`: number of hidden units
- `disable_progressbar = false`: set true if generating documentation
- `device = :cpu`: train using `:cpu` or `:gpu`
- `sort_in_nn = true`: (only for backend = "flux") whether put `sort` in `MLP` 
- `eval_in_batch = false`: (only for backend = "flux") Currently, `Flux` does not support `sort` in batch mode. A workaround with customized `Zygote.batch_sort` needs further verifications. 
"""
function ci_mono_ss_mlp(x::AbstractVector{T}, y::AbstractVector{T}, λs::AbstractVector{T}; 
                                                                        prop_nknots = 0.2,
                                                                        backend = "flux",
                                                                        model_file = "/tmp/model_G",
                                                                        nepoch0 = 3, nepoch = 3,
                                                                        niter_per_epoch = 100, # can be set via kw...
                                                                        M = 10,
                                                                        nhidden = 100,
                                                                        disable_progressbar = false,
                                                                        sort_in_nn = true, eval_in_batch = false, # only Flux
                                                                        device = :cpu, kw...) where T <: AbstractFloat
    B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
    model_file *= ifelse(backend == "flux", ".bson", ".pt")
    λl = minimum(λs)
    λu = maximum(λs)
    if backend == "flux"
        Ghat0, loss0 = train_Gλ(y, B, L; λl = λl, λu = λu,
                                         device = device, model_file = model_file, 
                                         niter_per_epoch = niter_per_epoch,
                                         sort_in_nn = sort_in_nn, eval_in_batch = eval_in_batch,
                                         nhidden = nhidden,
                                         disable_progressbar = disable_progressbar,
                                         nepoch = nepoch0, kw...)
        Ghat, loss = train_Gyλ(y, B, L, model_file; device = device, 
                                         nepoch = nepoch, niter_per_epoch = niter_per_epoch,
                                         λl = λl, λu = λu,
                                         sort_in_nn = sort_in_nn,
                                         M = M,
                                         disable_progressbar = disable_progressbar,
                                         kw...)
        LOSS = vcat(loss0, loss)                                         
    else
        Ghat, LOSS = py_train_G_lambda(y, B, L; nepoch = nepoch, nepoch0 = nepoch0, 
                                                gpu_id = ifelse(device == :cpu, -1, 0), 
                                                λl = λl, λu = λu,
                                                nhidden = nhidden,
                                                niter_per_epoch = niter_per_epoch,
                                                disable_tqdm = disable_progressbar,
                                                K = M, K0 = M, kw...)
    end
    Yhat = hcat([Ghat(y, λ) for λ in λs]...)
    if backend == "flux"
        Yhat0 = hcat([Ghat0(y, λ) for λ in λs]...)
    else
        Yhat0 = Yhat
    end
    RES_YCI, cov_hat = sample_G_λ(Ghat, y, λs)
    return Yhat, RES_YCI, LOSS, Yhat0
end

function sample_G_λ(G::Function, y::AbstractVector{T}, λs::AbstractVector{T}; 
                        nB = 100, α = 0.05) where T <: AbstractFloat
    n = length(y)
    nλ = length(λs)
    cov_hat = zeros(nλ, n)
    RES_YCI = Array{Any, 1}(undef, nλ)
    for (i, λ) in enumerate(λs)
        yhat = G(y, λ) 
        σhat = std(y - yhat)
        Ystar = hcat([yhat + randn(n) * σhat for _ in 1:nB]...)
        Yhat = hcat([G(Ystar[:, j], λ) for j in 1:nB]...)
        # idx = sortperm(x)
        YCI = hcat([quantile(t, [α/2, 1-α/2]) for t in eachrow(Yhat)]...)'
        RES_YCI[i] = YCI
        for j = 1:n
            cov_hat[i, j] = mean((Yhat[j, :] .- mean(Yhat[j, :])) .* (Ystar[j, :] .- mean(Ystar[j, :])) )
        end
    end
    return RES_YCI, cov_hat
end

"""
    py_train_G_lambda(y::AbstractVector, B::AbstractMatrix, L::AbstractMatrix; <keyword arguments>)

Wrapper for training MLP generator using PyTorch.

## Arguments

- `η0`, `η`: learning rate
- `K0`, `K`: Monte Carlo size
- `nepoch0`, `nepoch`: number of epoch
- `nhidden`, `depth`: size of MLP
- `λl`, `λu`: range of `λ`
- `use_torchsort = false`: `torch.sort` (default: false) or `torchsort.soft_sort` (true)
- `sort_reg_strength = 0.1`: tuning parameter when `use_torchsort = true`.
- `model_file`: path for saving trained model
- `gpu_id = 0`: use specified GPU
- `niter_per_epoch = 100`: number of iterations in each epoch
- `disable_tqdm = false`: set `true` when generating documentation
"""
function py_train_G_lambda(y::AbstractVector, B::AbstractMatrix, L::AbstractMatrix; 
                            η = 0.001, η0 = 0.001, 
                            K0 = 10, K = 10, 
                            nhidden = 1000, depth = 2,
                            γ = 0.9, # deprecated
                            decay_step = 5, # deprecated
                            patience = 100, patience0 = 100, disable_early_stopping = true, # deprecated
                            nepoch0 = 100, nepoch = 100, 
                            λl = 1e-9, λu = 1e-4,
                            use_torchsort = false, sort_reg_strength = 0.1,
                            model_file = "model_G.pt",
                            gpu_id = 0,
                            niter_per_epoch = 100,
                            disable_tqdm = false,
                            kw...
                            )
    Ghat, LOSS = _py_boot."train_G_lambda"(Float32.(y), Float32.(B), Float32.(L), eta = η, K = K, 
                                            K0 = K0,
                                            nepoch = nepoch,
                                            gamma = γ, eta0 = η0, 
                                            decay_step = decay_step, 
                                            nepoch0 = nepoch0, 
                                            lam_lo = λl, lam_up = λu, 
                                            model_file = model_file,
                                            use_torchsort = use_torchsort, sort_reg_strength=sort_reg_strength, 
                                            gpu_id = gpu_id, 
                                            patience0 = patience0, patience=patience, disable_early_stopping = disable_early_stopping,
                                            niter_per_epoch = niter_per_epoch,
                                            nhidden = nhidden, depth = depth,
                                            disable_tqdm = disable_tqdm)#::Tuple{PyObject, PyArray}
    #println(typeof(py_ret)) #Tuple{PyCall.PyObject, Matrix{Float32}} 
    # ....................... # Tuple{PyCall.PyObject, PyCall.PyArray{Float32, 2}}
    #LOSS = Matrix(py_ret[2]) # NB: not necessarily a matrix, but possibly a matrix
    return (y, λ) -> B * py"$Ghat"(Float32.(vcat(y, aug(λ)))), LOSS
    #return y -> py"$(py_ret[1])"(Float32.(y)), LOSS
end

"""
    load_model(n::Int, J::Int, nhidden::Int, model_file::String; dim_lam = 8, gpu_id = 3)

Load trained model from `model_file`.
"""
function load_model(B::Matrix, model_file::String; dim_lam = 8, gpu_id = 3)
    params = split_keystr(basename(model_file))
    n = params["n"]
    J = params["J"]
    nhidden = params["nhidden"]
    Ghat = _py_boot."load_model"(n, dim_lam, J, nhidden, model_file, gpu_id)
    return (y, λ) -> B * py"$Ghat"(Float32.(vcat(y, aug(λ))))
end

# adopt from https://github.com/szcf-weiya/Xfunc.jl/blob/0778903310bd9bc82880d55e640cd1888baaa599/src/str.jl#L31-L46
function split_keystr(x::String)
    xs = split(x, "-")
    res = Dict()
    for y in xs
        try
            ys = match(r"([a-zA-Z]+)(.*)", y).captures
            if ys[2] != ""
                res[ys[1]] = parse(Int, ys[2])
            end
        catch e
        end
    end
    return res
end