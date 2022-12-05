# rename to GpBS? (c.f. GBS)
using Flux
using Plots
using PyCall
using Serialization
using Random
using BSON

# currentdir = @__DIR__
# py"""
# import sys
# sys.path.insert(0, $(currentdir))
# from boot import train_G
# """

## determine functions formally (NB: Be better not to change the name)
cubic(x) = x^3
logit(x) = 1/(1+exp(-x))

function check_CI(;n = 100, σ = 0.1, f = exp, nB = 1000, nepoch = 200, 
                        K = 10, nrep = 100, α = 0.05, C = 1, η = 0.001, method = "pytorch",
                        K0 = 10,
                        λ = 0.1, cvλ = false, λs = exp.(range(-10, -4, length = 100)),
                        amsgrad = true,
                        γ = 0.9,
                        η0 = 0.0001,
                        demo = false,
                        max_norm = 2000.0, clip_ratio = 1000.0,
                        debug_with_y0 = false,
                        decay_step = round(Int, 1 / η),
                        nepoch0 = 100, N1 = 100, N2 = 100,
                        seed = 1234,
                        prop_nknots = 1.0,
                        nhidden = 1000, depth = 2,
                        gpu_id = 3,
                        model_file = nothing,
                        niter_per_epoch = 100, cooldown2 = 10,
                        patience = 100, cooldown = 100,
                        fig = true, figfolder = "~", kw...
                        )
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    nλ = length(λs)
    res_time = zeros(nrep)
    res_covprob = zeros(nrep) #
    res_overlap = zeros(nrep, nλ)
    res_err = zeros(nrep, 3)
    Err_boot = zeros(nrep, nλ, 3)
    if method == "lambda" || method == "lambda_from_file" || method == "jl_lambda"
        res_covprob = zeros(nrep, nλ)
        res_err = zeros(nrep, nλ, 3)
    end
    Yhat0 = zeros(nλ, n)
    Yhat = zeros(nλ, n)
    ## julia's progress bar has been overrideen by tqdm's progress bar
    for i = 1:nrep
        if nrep == 1
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed), n) * σ    
        else
            seed = Int(rand(UInt8))
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed), n) * σ    
            # x = rand(n) * 2 .- 1
            # y = f.(x) + randn(n) * σ    
        end
        B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
        if cvλ
            errs, _, _, _ = cv_mono_ss(x, y, λs)
            λopt = λs[argmin(errs)]
            βhat0, yhat0 = mono_ss(x, y, λopt)
            λ = λopt  / n # TODO: since LOSS + λopt Penalty, then LOSS / n + λ / n * Penalty
        else
            βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = prop_nknots);
        end
        for (j, λ) in enumerate(λs)
            βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = prop_nknots)
            Yhat0[j, :] = yhat0
        end
        # err0 = Flux.Losses.mse(yhat, y)
        σ0 = lm(x, y)
        res_time[i] = @elapsed begin
            if method == "pytorch"
                #Ghat = py_train_G(y, B, K = K, nepoch = nepoch, η = η, σ = σ0)
                Ghat = py_train_G(y, B, L, λ, K = K, nepoch = nepoch, η = η, σ = σ0, figname = ifelse(fig, "$figfolder/loss-$f-$i.png", nothing), amsgrad = amsgrad, γ = γ, η0 = η0, decay_step = decay_step, max_norm = max_norm, clip_ratio = clip_ratio,
                debug_with_y0 = debug_with_y0, y0 = f.(x), nepoch0 = nepoch0, N1 = N1, N2 = N2)
            elseif method == "jl_lambda"
                model_file = "model-$f-$σ-n$n-J$J-nhidden$nhidden-$i-$seed-$timestamp.bson"
                λl = λs[1]
                λu = λs[end]
                device = ifelse(gpu_id == -1, :cpu, :gpu)
                Ghat0, loss0 = train_Gλ(y, B, L; λl = λl, λu = λu,
                                                device = device, model_file = model_file, 
                                                niter_per_epoch = niter_per_epoch,
                                                nhidden = nhidden,
                                                M = K,
                                                nepoch = nepoch0, kw...)
                Ghat, loss = train_Gyλ(y, B, L, model_file; device = device, 
                                                nepoch = nepoch, niter_per_epoch = niter_per_epoch,
                                                M = K,
                                                λl = λl, λu = λu)
                LOSS = vcat(loss0, loss)
                if fig
                    savefig(plot(log.(LOSS)), "$figfolder/loss-$f-$σ-$i.png")
                end
            elseif method == "lambda"
                M = K
                Ghat, LOSS = py_train_G_lambda(y,
                                                B, L, K = M, nepoch = nepoch, η = η, 
                                                K0 = K0,
                                                figname = ifelse(fig, "$figfolder/loss-$f-$σ-$i.png", nothing), 
                                                amsgrad = amsgrad, γ = γ, η0 = η0, 
                                                use_torchsort = false,
                                                gpu_id = gpu_id,
                                                nhidden = nhidden, depth = depth,
                                                niter_per_epoch = niter_per_epoch, cooldown2 = cooldown2,
                                                model_file = "model-$f-$σ-n$n-J$J-nhidden$nhidden-$i-$seed-$timestamp.pt",
                                                patience = patience, cooldown = cooldown,
                                                decay_step = decay_step, max_norm = max_norm, clip_ratio = clip_ratio, 
                                                nepoch0 = nepoch0, λl = λs[1], λu = λs[end])
            elseif method == "lambda_from_file"
                # gpu is much faster
                Ghat = load_model(n, J, nhidden, model_file, gpu_id = gpu_id)
            else
                Ghat = train_G2(x, y, B; K = K, σ = σ0, nepoch = nepoch, nB = nB, C = C, patience = patience, η = η) 
            end
        end
        # Ghat = train_G2_batch(x, y, B; K = K, σ = σ0, nepoch = nepoch, nB = nB, C = C, patience = patience)
        # Flux.trainmode!(Ghat)
        # Flux.testmode!(Ghat)
        
        #Γhat, γhat, Γss = train_G(x, y, B; K = K, σ = σ0, nepoch = nepoch, nB = nB, C = C) 
        # CI, αs = double_boot(γhat, Γhat', Γss)
        # Yhat = B * Γhat # n x J x (J x nB) = n x nB
        # # CI = hcat([quantile(t, [α/2, 1-α/2]) for t in eachrow(Γhat)]...)'
        # # res[i] = coverage_prob(B * CI, f.(x)) 
        # YCI = hcat([quantile(t, [α/2, 1-α/2]) for t in eachrow(Yhat)]...)'
        # res[i] = coverage_prob(YCI, f.(x)) 
        if method == "lambda" || method == "lambda_from_file" || method == "jl_lambda"
            fitfig = scatter(x, y, legend=:topleft, label="", title="seed = $seed, J = $J")
            idx = sortperm(x)
            fit_err = zeros(nλ, n)
            RES_YCI0 = Array{Any, 1}(undef, nλ)
            for (j, λ) in enumerate(λs)
                _, YCI = MonotoneSplines.ci_mono_ss(x, y, λ, prop_nknots=prop_nknots)
                RES_YCI0[j] = YCI
                yhat = Ghat(y, λ)
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
            savefig(fitfig, "$figfolder/fit-$f-$σ-$i.png")
            RES_YCI, cov_hat = sample_G_λ(Ghat, y, λs, nB = nB)
            cp = [coverage_prob(YCI, f.(x)) for YCI in RES_YCI]
            if demo
                serialize("demo-CI-$f-n$n-σ$σ-seed$seed-B$nB-K0$K0-K$K-nepoch$nepoch-prop$(prop_nknots)-$timestamp.sil", [x, y, λs, J, Yhat, Yhat0, RES_YCI, RES_YCI0, cp]) # loss not exist when from file
            end
            for j = 1:nλ
                res_overlap[i, j] = jaccard_index(RES_YCI[j], RES_YCI0[j])
            end
            res_covprob[i, :] .= cp
            Err_boot[i, :, 1] .= mean(fit_err, dims=2)[:]
            Err_boot[i, :, 2] .= mean(cov_hat, dims=2)[:]
            Err_boot[i, :, 3] .= Err_boot[i, :, 1] + 2 * Err_boot[i, :, 2]
            errfig = plot(log.(λs), Err_boot[i, :, 3], label = "err + 2cov")
            plot!(log.(λs), Err_boot[i, :, 1], label = "err")
            errs, _, _, _ = cv_mono_ss(x, y, λs, nfold = 10)
            plot!(errfig, log.(λs), errs, label = "10 fold CV")
            errs2, _, _, _ = cv_mono_ss(x, y, λs, nfold = n)
            plot!(errfig, log.(λs), errs2, label = "LOOCV")
            savefig(errfig, "$figfolder/err-$f-$σ-$i.png")
            savefig(plot(log.(λs), cp), "$figfolder/cp-$f-$σ-$i.png")
            # sample_G_λ(Ghat, B, x, y, f, λs/n, nB = nB, figname = ifelse(fig, "$figfolder/fit-$f-$σ-$i.png", nothing))            
        else
            res_covprob[i], yhat = sample_G(Ghat, B, x, y, f, nB = nB, figname = ifelse(fig, "$figfolder/fit-$f-$i.png", nothing))
            res_err[i, :] = [Flux.Losses.mse(yhat0, yhat),
                            Flux.Losses.mse(yhat0, f.(x)),
                            Flux.Losses.mse(yhat, f.(x))
                            ]    
        end
        if fig
            if cvλ
                savefig(plot(log.(λs), log.(errs)), "$figfolder/cv-$f-$i.png")
                if strip(read(`hostname`, String)) == "chpc-gpu019"
                    run(`ssh sandbox convert $figfolder/cv-$f-$i.png $figfolder/fit-$f-$i.png $figfolder/loss-$f-$i.png +append $figfolder/$f-$i.png`)
                else
                    run(`convert $figfolder/cv-$f-$i.png $figfolder/fit-$f-$i.png $figfolder/loss-$f-$i.png +append $figfolder/$f-$i.png`)
                end
            else
                # run(`ssh sandbox convert $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
                if strip(read(`hostname`, String)) == "chpc-gpu019"
                    run(`ssh sandbox convert $figfolder/cp-$f-$σ-$i.png $figfolder/err-$f-$σ-$i.png $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
                else
                    run(`convert $figfolder/cp-$f-$σ-$i.png $figfolder/err-$f-$σ-$i.png $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
                end
            end
        end                    
    end
    serialize("$f-n$n-σ$σ-nrep$nrep-B$nB-K$K-λ$λ-η$η-nepoch$nepoch-$timestamp.sil", [res_covprob, res_overlap, res_err, res_time, Err_boot])
    return mean(res_overlap, dims=1)[1,:], mean(res_covprob, dims=1)[1,:], mean(res_err, dims=1)[1,:,:], mean(res_time)
end

function check_acc(; n = 100, σ = 0.1, f = exp, 
                        nrep = 10,
                        M = 10,
                        λs = exp.(range(-10, -4, length = 100)),
                        niter = 10000,
                        η0 = 1e-4, η = 1e-3,
                        nhidden = 1000, depth = 2,
                        seed = 1234,
                        use_torchsort = false, sort_reg_strength = 0.1,
                        gpu_id = 0,
                        patience = 100, cooldown = 100,
                        prop_nknots = 1.0,
                        demo = false,
                        max_norm = 2.0, clip_ratio = 1.0, decay_step=1000, amsgrad = true, γ = 1.0,
                        fig = true, figfolder = "~",
                        backend = "pytorch"
                        )
    timestamp = replace(strip(read(`date -Iseconds`, String)), ":" => "_")
    nλ = length(λs)
    res_time = zeros(nrep, 3)
    res_err = zeros(nrep, nλ, 3)
    Yhat0 = zeros(nλ, n)
    if demo & (nrep > 1)
        @warn "demo should work with nrep = 1"
    end
    if demo 
        Yhat = zeros(nλ, n)
    end
    for i = 1:nrep
        if nrep == 1
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed), n) * σ        
        else
            seed = Int(rand(UInt8))
            x = rand(MersenneTwister(seed), n) * 2 .- 1
            y = f.(x) + randn(MersenneTwister(seed), n) * σ    
            # x = rand(n) * 2 .- 1
            # y = f.(x) + randn(n) * σ        
        end
        # μy = mean(y)
        B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
        res_time[i, 1] = @elapsed for (j, λ) in enumerate(λs)
            βhat0, yhat0 = mono_ss(x, y, λ, prop_nknots = prop_nknots)
            Yhat0[j, :] = yhat0
        end
        res_time[i, 2] = @elapsed begin 
            if backend == "pytorch"
                Ghat, LOSS = py_train_G_lambda(y, #y .- μy, 
                                                                B, L, K = M, nepoch = 0, η = η, 
                                                                figname = ifelse(fig, "$figfolder/loss-$f-$σ-$i.png", nothing), 
                                                                amsgrad = amsgrad, γ = γ, η0 = η0, 
                                                                use_torchsort = use_torchsort,
                                                                sort_reg_strength = sort_reg_strength,
                                                                gpu_id = gpu_id,
                                                                nhidden = nhidden, depth = depth,
                                                                patience = patience, cooldown = cooldown,
                                                                decay_step = decay_step, max_norm = max_norm, clip_ratio = clip_ratio, nepoch0 = niter, λl = λs[1], λu = λs[end])
            else
                Ghat, LOSS = train_G(y, B, L)
            end
        end
        if fig
            fitfig = scatter(x, y, legend=:topleft, label="", title="seed = $seed, J = $J")
            idx = sortperm(x)
        end
        res_time[i, 3] = @elapsed for (j, λ) in enumerate(λs)
            yhat = Ghat(y, λ)
            if demo
                Yhat[j, :] = yhat
            end
            rel_gap = Flux.Losses.mse(Yhat0[j, :], yhat) / Flux.Losses.mse(Yhat0[j, :], zeros(n))
            fit_ratio = Flux.Losses.mse(yhat, y) / Flux.Losses.mse(Yhat0[j, :], y)
            fit_ratio2 = Flux.Losses.mse(yhat, f.(x)) / Flux.Losses.mse(Yhat0[j, :], f.(x))
            res_err[i, j, :] .= [rel_gap, fit_ratio, fit_ratio2]
            if fig
                plot!(fitfig, x[idx], yhat[idx], label = "λ = $λ")
                plot!(fitfig, x[idx], Yhat0[j, idx], label = "", ls = :dash)
            end
        end
        if fig
            savefig(fitfig, "$figfolder/fit-$f-$σ-$i.png")
            if strip(read(`hostname`, String)) == "chpc-gpu019"
                run(`ssh sandbox convert $figfolder/fit-$f-$σ-$i.png $figfolder/loss-$f-$σ-$i.png +append $figfolder/$f-$σ-$i.png`)
            else # on rocky
                run(`convert $figfolder/fit-$σ-$f-$i.png $figfolder/loss-$σ-$f-$i.png +append $figfolder/$f-$σ-$i.png`)
            end
        end
        if demo
            serialize("demo-acc-$f-n$n-σ$σ-seed$seed-M$M-η0$η0-niter$niter-prop$(prop_nknots)-$timestamp.sil", [x, y, λs, J, Yhat, Yhat0, LOSS])
        end
    end
    serialize("acc-$f-n$n-σ$σ-nrep$nrep-M$M-η0$η0-niter$niter-$timestamp.sil", [res_err, res_time])
    return mean(res_err, dims=1)[1,:,:]#, mean(res_time)
end

"""
    aug(λ)

Augment `λ` with 8 different functions.
"""
aug(λ::AbstractFloat) = [λ, cbrt(λ), exp(λ), sqrt(λ), log(λ), 10*λ, λ^2, λ^3]

"""
    train_G(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix; λl, λu)

Train MLP generator G(λ) for λ ∈ [λl, λu].
"""
function train_Gλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix; dim_λ = 8, 
                                        activation = gelu,
                                        nhidden = 100,
                                        M = 10,
                                        niter_per_epoch = 100, nepoch = 3,
                                        device = :cpu,
                                        model_file = "/tmp/model_G.bson",
                                        λl = 1e-4, λu = 1e-3, kw...)
    device = eval(device)
    y = device(rawy)
    B = device(rawB)
    L = device(rawL)
    n, J = size(B)
    G = Chain(Dense(n + dim_λ => nhidden, activation), 
        Dense(nhidden => nhidden, activation),
        Dense(nhidden => nhidden, activation), 
        Dense(nhidden => J),
        sort
    ) |> device
    opt = AMSGrad()
    loss(λ::AbstractFloat) = Flux.Losses.mse(B * G(vcat(y, aug(λ))), y) + λ * sum((L' * G(vcat(y, aug(λ))) ).^2) / n
    losses(λs::Vector) = mean([loss(λ) for λ in λs])
    train_loss = Float64[]
    for epoch in 1:nepoch
        for i in 1:niter_per_epoch
            λs = rand(M) * (λu - λl) .+ λl
            Flux.train!(losses, Flux.params(G), [λs], opt)
            append!(train_loss, losses(λs))
        end
    end
    G = cpu(G)
    BSON.@save model_file G
    return (y, λ) -> B * G(vcat(y, aug(λ))), train_loss
end

function train_Gyλ(rawy::AbstractVector, rawB::AbstractMatrix, rawL::AbstractMatrix, model_file::String; device = :cpu, 
                        niter_per_epoch = 100, nepoch = 3, λl = 1e-4, λu = 1e-3,
                        M = 10)
    device = eval(device)
    y = device(rawy)
    B = device(rawB)
    L = device(rawL)
    n, J = size(B)
    # G0 = BSON.@load model_file G (see issue #3)
    # G = BSON.@load model_file G
    G0 = BSON.load(model_file, @__MODULE__)[:G]
    G = BSON.load(model_file, @__MODULE__)[:G]
    G0 = device(G0)
    G = device(G)
    opt = AMSGrad()
    train_loss = Float64[]
    function loss(λ::AbstractFloat)
        ypred = B * G0(vcat(y, aug(λ)))
        σ = std(ypred - y)
        ytrain = [ypred + randn(n) * σ for _ in 1:M]
        return mean([Flux.Losses.mse(B * G(vcat(ytrain[i], aug(λ))), ytrain[i]) + λ * sum((L' * G(vcat(ytrain[i], aug(λ))) ).^2) / n for i = 1:M])
    end
    losses(λs::Vector) = mean([loss(λ) for λ in λs])
    for epoch in 1:nepoch
        for i in 1:niter_per_epoch
            λs = rand(M) * (λu - λl) .+ λl
            Flux.train!(losses, Flux.params(G), [λs], opt)
            append!(train_loss, losses(λs))
        end
    end
    G = cpu(G)
    model_file1 = model_file[1:end-5] * "_ci.bson" # keep G0
    BSON.@save model_file1 G
    return (y, λ) -> B * G(vcat(y, aug(λ))), train_loss
end



"""
    mono_ss_mlp(x::AbstractVector, y::AbstractVector; λl, λu)

Fit monotone smoothing spline by training a MLP generator.
"""
function mono_ss_mlp(x::AbstractVector, y::AbstractVector; prop_nknots = 0.2, backend = "flux", λl = 1e-5, λu = 1e-4, device = :cpu, kw...)
    B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
    if backend == "flux"
        Ghat, LOSS = train_Gλ(y, B, L; λl = λl, λu = λu, device = device, kw...)
    else
        Ghat, LOSS = py_train_G_lambda(y, B, L; nepoch = 0, nepoch0 = 3, 
                                                λl = λl, λu = λu, 
                                                gpu_id = ifelse(device == :cpu, -1, 0), kw...)
    end
    return Ghat, LOSS
end

"""
    ci_mono_ss_mlp(x::AbstractVector{T}, y::AbstractVector{T}, λs::AbstractVector{T}; )

Fit data `x, y` at each `λs` with confidence bands.
"""
function ci_mono_ss_mlp(x::AbstractVector{T}, y::AbstractVector{T}, λs::AbstractVector{T}; 
                                                                        prop_nknots = 0.2,
                                                                        backend = "flux",
                                                                        model_file = "/tmp/model_G",
                                                                        nepoch0 = 3, nepoch = 3,
                                                                        niter_per_epoch = 100, # can be set via kw...
                                                                        M = 10,
                                                                        nhidden = 100,
                                                                        device = :cpu, kw...) where T <: AbstractFloat
    B, Bnew, L, J = build_model(x, true, prop_nknots = prop_nknots)
    model_file *= ifelse(backend == "flux", ".bson", ".pt")
    λl = minimum(λs)
    λu = maximum(λs)
    if backend == "flux"
        Ghat0, loss0 = train_Gλ(y, B, L; λl = λl, λu = λu,
                                         device = device, model_file = model_file, 
                                         niter_per_epoch = niter_per_epoch,
                                         nhidden = nhidden,
                                         nepoch = nepoch0, kw...)
        Ghat, loss = train_Gyλ(y, B, L, model_file; device = device, 
                                         nepoch = nepoch, niter_per_epoch = niter_per_epoch,
                                         λl = λl, λu = λu,
                                         kw...)
        LOSS = vcat(loss0, loss)                                         
    else
        Ghat, LOSS = py_train_G_lambda(y, B, L; nepoch = nepoch, nepoch0 = nepoch0, 
                                                gpu_id = ifelse(device == :cpu, -1, 0), 
                                                λl = λl, λu = λu,
                                                nhidden = nhidden,
                                                niter_per_epoch = niter_per_epoch,
                                                K = M, K0 = M, kw...)
    end
    Yhat = hcat([Ghat(y, λ) for λ in λs]...)
    RES_YCI, cov_hat = sample_G_λ(Ghat, y, λs)
    return Yhat, RES_YCI, LOSS
end

function train_G(rawx, rawy, rawB, rawL, λ;  nepoch = 100, σ = 0.1, K = 10, 
                                            nhidden = 1000, η = 0.001, nB = 100, 
                                            device = gpu,
                                            f = x->x^3,
                                            debug_with_y0 = false,
                                            patience = 10)
    x = device(rawx)
    y = device(rawy)
    B = device(rawB)
    L = device(rawL)
    n, J = size(B)
    G = Chain(Dense(n => nhidden, relu), 
        Dense(nhidden => nhidden, relu),
        Dense(nhidden => nhidden, relu), 
        Dense(nhidden => J), # the last layer cannot be relu
        sort
    ) |> device
    loss1(y) = Flux.Losses.mse(B * G(y), y) + λ * sum((L' * G(y)).^2)
    loss2(yhat, εs) = mean([Flux.Losses.mse(yhat + ε, B * G(yhat + ε) ) + λ * sum((L' * G(yhat + ε)).^2) for ε in εs])
    opt1 = Adam(η / K)
    opt2 = Adam(η)
    LOSS = zeros(nepoch, 2)
    for epoch in 1:nepoch
        Flux.train!(loss1, Flux.params(G), [y], opt1)
        yhat = B * G(y)
        σ = std(y - yhat)
        εs = [device(randn(n) * σ) for _ in 1:K]
        Flux.train!(loss2, Flux.params(G), [(yhat, εs)], opt2)
        LOSS[epoch, :] .= [loss2(yhat, εs), loss1(y)]
        println("epoch = $epoch, loss(y) = $(LOSS[epoch, :]), sigma = $σ")
    end
    savefig(plot(log.(LOSS)), "loss.png")
    return cpu(G)
end

function train_G2(rawx, rawy, rawB; nepoch = 100, σ = 0.1, K = 10, 
                                    nhidden = 1000, η = 0.001, nB = 100, 
                                    device = gpu, C = 100,
                                    f = x->x^3,
                                    debug_with_y0 = false,
                                    patience = 10)
    x = device(rawx)
    y0 = f.(x)
    y = device(rawy)
    B = device(rawB)
    n = length(y)
    J = size(B, 2)
    σ0 = σ
    G = Chain(Dense(n => nhidden, relu), 
              Dense(nhidden => nhidden, relu),
              Dense(nhidden => nhidden, relu), 
              Dense(nhidden => J), # the last layer cannot be relu
              sort
              ) |> device
    # loss(x, y) = mean([Flux.Losses.mse(B * G(yi), yi) for yi in y])
    loss1(y) = Flux.Losses.mse(B * G(y), y)
    # loss2(yhat, εs) = mean([Flux.Losses.mse(yhat + ε, B * G(yhat + ε) ) - ε' * ε / n for ε in εs].^2)
    loss2(yhat, εs) = mean([Flux.Losses.mse(yhat + ε, B * G(yhat + ε) ) for ε in εs])
    # loss2(yhat, εs) = mean([Flux.Losses.huber_loss(yhat + ε, B * G(yhat + ε) ) for ε in εs])
    loss(y, εs) = mean([Flux.Losses.mse(B * G(y) + ε, B * G(B * G(y) + ε) ) for ε in εs])
    #loss_all(y, εs) = (loss(y, εs) * K + loss(y, [y - B * G(y)])) / (1 + K)
    loss_all(y, εs) = (loss(y, εs) + loss(y, [y - B * G(y)])) / 2
    opt = AMSGrad()
    opt1 = Adam(η/K)
    opt2 = Adam(η)
    # opt = RAdam(η)
    LOSS = zeros(nepoch, 2)
    # init_loss = loss(x, [y])  # NB: no x!!
    init_loss = loss(y, [y - B * G(y)])
    println("init loss = ", init_loss)
    # es1 = Flux.early_stopping(loss1, patience)
    # es2 = Flux.early_stopping(loss2, patience)
    # es1(i) = Flux.early_stopping(i->LOSS[i, 2], patience)
    # es2(i) = Flux.early_stopping(i->LOSS[i, 1], patience)
    wait = [0, 0]
    best_loss = [Inf, Inf]
    Gold = nothing
    for epoch in 1:nepoch
        # ystar = y + randn(n) * σ
        #Flux.train!(loss, Flux.params(G), [(x, ystar)], opt)
        # ytrain = [y0 + device(randn(n) * σ) for _ in 1:K]
        # Flux.train!(loss, Flux.params(G), [(x, ytrain)], opt)
        #Flux.train!(loss, Flux.params(G), [(y, εs)], opt)
        # Flux.train!(loss_all, Flux.params(G), [(y, εs)], opt)
        if debug_with_y0
            σ = std(y - y0)
        else
            Flux.train!(loss1, Flux.params(G), [y], opt1)
            yhat = B * G(y)
            σ = std(y - yhat)
        end
        εs = [device(randn(n) * σ) for _ in 1:K]
        # ytrain = [y0 + ε for ε in εs]
        if debug_with_y0
            Flux.train!(loss2, Flux.params(G), [(y0, εs)], opt)
            LOSS[epoch, :] .= [loss2(y0, εs), loss1(y)]
        else
            Flux.train!(loss2, Flux.params(G), [(yhat, εs)], opt2)
            LOSS[epoch, :] .= [loss2(yhat, εs), loss1(y)]
        end
        # y_batch = reduce(hcat, ytrain) # n x K
        # println(size(y_batch))
        # println(size(G(y_batch))) # p x K
        # Flux.train!(loss2b, Flux.params(G), [y_batch], opt)
        # LOSS[epoch, :] .= [loss(x, ytrain), loss(x , [y])]
        #LOSS[epoch, :] .= [loss(y, εs), loss(y , [device(zeros(n))])]
        #LOSS[epoch, :] .= [loss(y, εs), loss(y , [y - B*G(y)])]
        ## TODO: concern, is it possible one goes down and another goes up, so should NOT use &&
        ## but this go up and go down just appear in another version train_G 
        # if es1(epoch) && es2(epoch)
        #     savefig(plot(LOSS[1:epoch, :]), "loss.png")
        #     return cpu(G) # necessary to record the best model and then return it?
        # end
        for i = 1:2
            if LOSS[epoch, i] < best_loss[i]
                best_loss[i] = LOSS[epoch, i]
                wait[i] = 0
                if i == 2
                    Gold = cpu(G)
                end
            else
                wait[i] += 1
            end
        end
        # also should enable sufficient fitting
        if (minimum(wait) > patience) # && σ < σ0
            @info "Early Stopped!"
            savefig(plot(LOSS[1:epoch, :]), "loss.png")
            # return Gold
            break
        end        
        println("epoch = $epoch, loss(y) = $(LOSS[epoch, :]), sigma = $σ")
        # println("G(y) = ", G(y)')
    end
    savefig(plot(log.(LOSS)), "loss.png")
    return cpu(G)
end

function train_G(rawx, rawy, rawB; nepoch = 100, σ = 0.1, K = 10, nhidden = 1000, η = 0.001, nB = 100, device = gpu, C = 100, patience = 3, K2=100, f = x->x^3)
    x = device(rawx)
    y0 = f.(x)
    y = device(rawy)
    B = device(rawB)
    n = length(y)
    G = Chain(Dense(n => nhidden, relu), 
              Dense(nhidden => nhidden, relu),
              Dense(nhidden => nhidden, relu), 
              Dense(nhidden => J), # the last layer cannot be relu
              sort
              ) |> device
    # loss(x, y) = mean([Flux.Losses.mse(B * G(yi), yi) for yi in y])
    loss1(y) = Flux.Losses.mse(B * G(y), y)
    loss2(yhat, εs) = mean([Flux.Losses.mse(yhat + ε, B * G(yhat + ε) ) for ε in εs])
    loss(y, εs) = mean([Flux.Losses.mse(B * G(y) + ε, B * G(B * G(y) + ε) ) for ε in εs])
    loss_all(y, εs) = (loss(y, εs) * K + loss(y, [y - B * G(y)])) / (1 + K)
    # loss_all(y, εs) = (loss(y, εs) + loss(y, [y - B * G(y)])) / 2
    # opt = AMSGrad()
    # opt = Adam(η)
    opt = RAdam(η)
    LOSS = zeros(nepoch, 2)
    # init_loss = loss(x, [y])  # NB: no x!!
    init_loss = loss(y, [y - B * G(y)])
    println("init loss = ", init_loss)
    Gold = nothing
    smallest_gap = Inf
    wait = [0, 0]
    best_loss = [Inf, Inf]
    for epoch in 1:nepoch
        # ystar = y + randn(n) * σ
        #Flux.train!(loss, Flux.params(G), [(x, ystar)], opt)
        # ytrain = [y0 + device(randn(n) * σ) for _ in 1:K]
        εs = [device(randn(n) * σ) for _ in 1:K]
        # Flux.train!(loss, Flux.params(G), [(x, ytrain)], opt)
        #Flux.train!(loss, Flux.params(G), [(y, εs)], opt)
        Flux.train!(loss_all, Flux.params(G), [(y, εs)], opt)
        # LOSS[epoch, :] .= [loss(x, ytrain), loss(x , [y])]
        #LOSS[epoch, :] .= [loss(y, εs), loss(y , [device(zeros(n))])]
        #LOSS[epoch, :] .= [loss(y, εs), loss(y , [y - B*G(y)])]
        LOSS[epoch, :] .= [loss_all(y, εs), loss(y , [y - B*G(y)])]
        for i = 1:2
            if LOSS[epoch, i] < best_loss[i]
                best_loss[i] = LOSS[epoch, i]
                wait[i] = 0
                if i == 2
                    Gold = cpu(G)
                end
            else
                wait[i] += 1
            end
        end
        if maximum(wait) > patience
            @info "Early Stopped!"
            savefig(plot(LOSS[1:epoch, :]), "loss.png")
            # return Gold
            break
        end
        # if epoch > patience
        #     gap = abs.(LOSS[epoch-patience:epoch, 2] - LOSS[epoch-patience:epoch, 1])
        #     if gap[end] < smallest_gap
        #         smallest_gap = gap[end]
        #         Gold = cpu(G)
        #     end
        #     if early_stop(gap)
        #         @info "Early Stopped!"
        #         savefig(plot(LOSS[1:epoch, :]), "loss.png")
        #         return Gold
        #     end
        # end
        σnew = std(y - B * G(y))
        println("epoch = $epoch, loss(y) = $(LOSS[epoch, :]), sigma = $σ, sigmanew = $σnew ")
        if σnew < σ
            σ = σnew
        end
        # println("G(y) = ", G(y)')
    end
    savefig(plot(LOSS), "loss.png")
    return cpu(G)
end

function sample_G(G, B::AbstractMatrix{T}, x::AbstractVector{T}, y::AbstractVector{T}, f::Function; 
                        device = cpu, nB = 100, α = 0.05, figname = "cubic.png") where T <: AbstractFloat
    γhat = G(y) # stupid bug!!!!! always use wrong y!!!
    yhat = B * γhat
    σhat = std(y - yhat)
    n = length(y)
    Γhat = hcat([G(B * γhat + device(randn(n) * σhat) ) for _ in 1:nB]...)
    Yhat = B * Γhat
    idx = sortperm(x)
    YCI = hcat([quantile(t, [α/2, 1-α/2]) for t in eachrow(Yhat)]...)'
    cp = coverage_prob(YCI, f.(x))
    if !isnothing(figname)
        plot(x[idx], Yhat[idx,:], legend = false, title = "B = $nB, Cov = $cp")
        scatter!(x, y)
        plot!(x[idx], yhat[idx], lw = 2, markershape=:star5)
        plot!(x[idx], YCI[idx,:], lw = 2)
        plot!(x[idx], f.(x[idx]), ls = :dash, lw = 2)
        savefig(figname)
    end
    return cp, yhat
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


# https://github.com/szcf-weiya/CrossValidation/blob/e4a7c41538842434accc32d1d9f71fc4ed4d5278/src/sim.jl#L979
function lm(x::AbstractVector, y::AbstractVector)
    μ = mean(x)
    k = sum( (x .- μ) .* y ) / sum( (x .- μ).^2 )
    k0 = mean(y) - k * μ
    return std(y - k * x .- k0)
end


function py_train_G(y::AbstractVector, B::AbstractMatrix; η = 0.001, K = 10, nepoch = 100, σ = 1.0)
    # Ghat, LOSS1, LOSS2 = py"train_G"(Float32.(y), Float32.(B), eta = η, K = K, nepoch = nepoch, sigma = σ)
    #Ghat, LOSS = py"train_G"(Float32.(y), Float32.(B), eta = η, K = K, nepoch = nepoch, sigma = σ)
    Ghat, LOSS = _py_boot."train_G"(Float32.(y), Float32.(B), eta = η, K = K, nepoch = nepoch, sigma = σ)
    # savefig(plot(
    #     # plot(log.(LOSS1)),
    #     # plot(log.(LOSS2)),
    # ), "pyloss.png")
    savefig(plot(log.(LOSS)), "pyloss.png")
    return y -> py"$Ghat"(Float32.(y))
    # if return as a lambda function, just use one dollar
end

function py_train_G(y::AbstractVector, B::AbstractMatrix, L::AbstractMatrix, λ::AbstractFloat; 
                            η = 0.001, η0 = 0.001, K = 10, nepoch = 100, σ = 1.0, 
                            amsgrad = true,
                            γ = 0.9,
                            max_norm = 2.0, clip_ratio = 1.0,
                            decay_step = Int(1 / η),
                            debug_with_y0 = false, y0 = 0, 
                            nepoch0 = 100, N1 = 100, N2 = 100,
                            figname = "pyloss.png" # not plot if nothing
                            )
    # Ghat, LOSS1, LOSS2 = py"train_G"(Float32.(y), Float32.(B), eta = η, K = K, nepoch = nepoch, sigma = σ)
    # Ghat, LOSS = py"train_G"(Float32.(y), Float32.(B), Float32.(L), λ, eta = η, K = K, nepoch = nepoch, sigma = σ)
    Ghat, LOSS = _py_boot."train_G"(Float32.(y), Float32.(B), Float32.(L), λ, eta = η, K = K, nepoch = nepoch, sigma = σ, amsgrad = amsgrad, gamma = γ, eta0 = η0, decay_step = decay_step, max_norm = max_norm, clip_ratio = clip_ratio, debug_with_y0 = debug_with_y0, y0 = Float32.(y0), nepoch0 = nepoch0, N1 = N1, N2 = N2)
    # savefig(plot(
    #     # plot(log.(LOSS1)),
    #     # plot(log.(LOSS2)),
    # ), "pyloss.png")
    if !isnothing(figname)
        savefig(plot(log.(LOSS),title = "λ = $λ"), figname)
    end
    return y -> py"$Ghat"(Float32.(y))
end

function py_train_G_lambda(y::AbstractVector, B::AbstractMatrix, L::AbstractMatrix; 
                            η = 0.001, η0 = 0.001, K = 10, nepoch = 100, σ = 1.0, 
                            K0 = 10,
                            nhidden = 1000, depth = 2,
                            amsgrad = true,
                            γ = 0.9,
                            max_norm = 2.0, clip_ratio = 1.0,
                            decay_step = Int(1 / η),
                            patience = 100, patience0 = 100, disable_early_stopping = true,
                            debug_with_y0 = false, y0 = 0, 
                            nepoch0 = 100,
                            λl = 1e-9, λu = 1e-4,
                            use_torchsort = false,
                            sort_reg_strength = 0.1,
                            model_file = "model_G.pt",
                            gpu_id = 0,
                            niter_per_epoch = 100,
                            figname = "pyloss.png", # not plot if nothing
                            kw...
                            )
    # Ghat, LOSS1, LOSS2 = py"train_G"(Float32.(y), Float32.(B), eta = η, K = K, nepoch = nepoch, sigma = σ)
    # Ghat, LOSS = py"train_G"(Float32.(y), Float32.(B), Float32.(L), λ, eta = η, K = K, nepoch = nepoch, sigma = σ)
    Ghat, LOSS = _py_boot."train_G_lambda"(Float32.(y), Float32.(B), Float32.(L), eta = η, K = K, 
                                            K0 = K0,
                                            nepoch = nepoch, sigma = σ, amsgrad = amsgrad, 
                                            gamma = γ, eta0 = η0, 
                                            decay_step = decay_step, max_norm = max_norm, clip_ratio = clip_ratio, 
                                            debug_with_y0 = debug_with_y0, y0 = Float32.(y0), 
                                            nepoch0 = nepoch0, 
                                            lam_lo = λl, lam_up = λu, 
                                            model_file = model_file,
                                            use_torchsort = use_torchsort, sort_reg_strength=sort_reg_strength, 
                                            gpu_id = gpu_id, 
                                            patience0 = patience0, patience=patience, disable_early_stopping = disable_early_stopping,
                                            niter_per_epoch = niter_per_epoch,
                                            nhidden = nhidden, depth = depth)#::Tuple{PyObject, PyArray}
    #println(typeof(py_ret)) #Tuple{PyCall.PyObject, Matrix{Float32}} 
    # ....................... # Tuple{PyCall.PyObject, PyCall.PyArray{Float32, 2}}
    #LOSS = Matrix(py_ret[2]) # NB: not necessarily a matrix, but possibly a matrix
    if !isnothing(figname)
        savefig(plot(log.(LOSS)), figname)
    end
    return (y, λ) -> B * py"$Ghat"(Float32.(vcat(y, aug(λ)))), LOSS
    #return y -> py"$(py_ret[1])"(Float32.(y)), LOSS
end

"""
    load_model(n::Int, J::Int, nhidden::Int, model_file::String; dim_lam = 8, gpu_id = 3)

Load trained model from `model_file`.
"""
function load_model(model_file::String; dim_lam = 8, gpu_id = 3)
    params = split_keystr(basename(model_file))
    n = params["n"]
    J = params["J"]
    nhidden = params["nhidden"]
    Ghat = _py_boot."load_model"(n, dim_lam, J, nhidden, model_file, gpu_id)
    return (y, λ) -> B * py"$Ghat"(Float32.(vcat(y, aug(λ))))
end

# deprecated
function load_model(n::Int, J::Int, nhidden::Int, model_file::String; dim_lam = 8, gpu_id = 3)
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