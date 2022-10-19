using LinearAlgebra
using RCall
using JuMP
using ECOS
using Plots
using StatsBase

OPTIMIZER = ECOS.Optimizer

"""
This is a `Spl`

# Fields
- `H`: an `RObject` generated by `splines::bs()`
- `β`: the coefficients for the B-spline.
"""
mutable struct Spl{T <: AbstractFloat}
    # H::Matrix{T}
    H::RObject{RealSxp}
    β::AbstractVector{T}
end

function monotone_spline(X, y, paras)
    return fit(X, y, paras, "monotone")
end

function bspline(X, y, J)
    return fit(X, y, J, "normal")
end

"""
    fit(X, y, paras, method)

`paras` is either the number of basis functions, or the sequence of interior knots. Return a `Spl` object.

```julia
n = 100
x = rand(n) * 2 .- 1
y = x .^3 + randn(n) * 0.01
res = fit(x, y, 10, "monotone")
```
"""
function fit(X::AbstractVector, y::AbstractVector, paras, method = "monotone", type = "increasing")
    if isa(paras, Int)
        J = paras
    # for cubic, order = 4, then take length of knots as J-4
#    H = R"splines::bs($X, knots = exp(seq(-4, 0, length=$J-4)), df=$J, intercept=TRUE, Boundary.knots = c(0, 1))"
        H = R"splines::bs($X, df=$J, intercept=TRUE)"
    else #isa(paras, Array)
        J = length(paras) + 4
        H = R"splines::bs($X, knots=$paras, intercept=TRUE)"
    end
    if method == "monotone"
        A = diagm(0 => ones(J), 1 => -ones(J-1))[1:J-1,:] # β1 - β2 >= 0 => decreasing
        if type == "increasing"
            A = -A
        end
        b = zeros(J-1)
        β = rcopy(R"lsei::lsi($H, $y, e = $A, f = $b)")
    else
        fm = R"lm($y ~ 0 + $H)"
        β = rcopy(R"$fm$coefficients")
    end
    # convert missing to 0 (issue #2)
    flag_missing = ismissing.(β)
    β[flag_missing] .= 0
    # return H, β
    return Spl(H, Float64.(β))
end

"""
    predict(model::Spl{T}, xs::AbstractVector{T})
    predict(X::Vector{Float64}, y::Vector{Float64}, J::Int, Xnew::AbstractVector{Float64}, ynew::AbstractVector{Float64}
    predict(X::Vector{Float64}, y::Vector{Float64}, J::Int, Xnew::Vector{Float64}, ynew::Vector{Float64}, σ::Vector{Float64}

Make prediction based on fitted `Spl` on new points `xs`. If `Xnew` is provided, then also returns the prediction error `‖yhat - ynew‖_2^2`.
"""
function predict(model::Spl{T}, xs::AbstractVector{T}) where T <: AbstractFloat
    return rcopy(R"predict($(model.H), $xs)") * model.β
end

function predict(X::Vector{Float64}, y::Vector{Float64}, J::Int,
                 Xnew::AbstractVector{Float64}, ynew::AbstractVector{Float64}, method = "monotone", tol = 1e-6)
    model = fit(X, y, J, method)
    H, β = model.H, model.β
    yhat = predict(model, Xnew)
    return yhat, sum((yhat - ynew) .^2), sum(abs.(β[1:J-1] - β[2:J]) .< tol) # number of equal pairs
end
function predict(X::Vector{Float64}, y::Vector{Float64}, J::Int,
                 Xnew::Vector{Float64}, ynew::Vector{Float64}, σ::Vector{Float64}, method = "monotone", xgrid = nothing)
    model = fit(X, y, J, method)
    yhat = predict(model, Xnew)
    if isnothing(xgrid)
        return yhat, sum(((yhat - ynew) ./σ) .^2)
    else
        ygrid = predict(model, xgrid)
        return yhat, sum(((yhat - ynew) ./σ) .^2), ygrid
    end
end


function cv_err(X, y, nfold = 10, J = 10)
    n = length(y)
    folds = div_into_folds(n, K = nfold)
    err = zeros(nfold)
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        spl = fit(X[train_idx], y[train_idx], J, "monotone", "increasing")
        yhat = predict(spl, X[test_idx])
        err[k] = norm(yhat - y[test_idx])^2
    end
    return sum(err) / n
end

"""
    eval_penalty(model::Spl{T}, x::AbstractVector{T})

Evaluate the penalty matrix by R's `fda::eval.penalty`. To make sure the corresponding design matrix contructed by `fda::eval.basis` is the same as `model.H`, it asserts the norm difference should be smaller than `sqrt(eps())`.
"""
function eval_penalty(model::Spl{T}, x::AbstractVector{T}) where T <: AbstractFloat
    knots = R"attr($(model.H), 'knots')"
    bd = R"attr($(model.H), 'Boundary.knots')"
    bbasis = R"fda::create.bspline.basis(breaks = c($(bd[1]), $knots, $(bd[2])), norder = 4)"
    # check B matrix is the same
    B = rcopy(R"fda::eval.basis($x, $bbasis)")
    @assert norm(B - rcopy(model.H)) < sqrt(eps())
    return rcopy(R"fda::eval.penalty($bbasis, 2)")
end

function mono_ss(x::AbstractVector, y::AbstractVector, λ = 1.0; ε = (eps())^(1/3))
    B, Bnew, L, J = build_model(x, true)
    return mono_ss(B, y, L, J, λ; ε = ε)
end

# shared B
"""
    mono_ss(B::AbstractMatrix, y::AbstractVector, L::AbstractMatrix, J::Int, λ::AbstractFloat)

Monotone Fitting with Smoothing Splines given design matrix `B` and cholesky-decomposed matrix `L`.
"""
function mono_ss(B::AbstractMatrix, y::AbstractVector, L::AbstractMatrix, J::Int, λ = 1.0; ε = (eps())^(1/3))
    A = zeros(Int, J-1, J)
    for i = 1:J-1
        A[i, i] = 1
        A[i, i+1] = -1
    end
    
    model = Model(OPTIMIZER)
    # if BarHomogeneous
    #     set_optimizer_attribute(model, "BarHomogeneous", 1)
    # end
    set_silent(model)
    @variable(model, β[1:J])
    @variable(model, z)
    @constraint(model, c1, A * β .<= 0)
    @constraint(model, c2, [z; vcat(y - B * β, sqrt(λ) * L' * β)] in SecondOrderCone())
    @objective(model, Min, z)
    optimize!(model)
    status = termination_status(model)
    if status in [MOI.OPTIMAL, MOI.ALMOST_OPTIMAL]
        βhat = value.(β)
        yhat = B * βhat
        return βhat, yhat, B, L
    else
        println(status)
        return nothing, mean(y) * ones(length(y))
    end
end

"""
    cv_mono_ss(x::AbstractVector{T}, y::AbstractVector{T}, λs::AbstractVector{T})

Cross-validation for monotone fitting with smoothing spline on `y ~ x` among parameters `λs`.
"""
function cv_mono_ss(x::AbstractVector{T}, y::AbstractVector{T}, λs = exp.(-6:0.5:1); ε = (eps())^(1/3)) where T <: AbstractFloat
    B, Bnew, L, J = build_model(x, true)

    nfold = 10
    n = length(y)
    folds = div_into_folds(n, K = nfold, seed = -1)
    nλ = length(λs)
    err = zeros(nfold, nλ)
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        for (i, λ) in enumerate(λs)
            βhat, _ = mono_ss(B[train_idx, :], y[train_idx], L, J, λ)
            err[k, i] = norm(B[test_idx, :] * βhat - y[test_idx])^2
        end
    end
    return sum(err, dims = 1)[:] / n, B, L, J
end
