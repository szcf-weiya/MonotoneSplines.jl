using LinearAlgebra
using RCall
using JuMP
using ECOS
using Plots
using StatsBase
import StatsBase.predict
import RCall.rcopy

OPTIMIZER = ECOS.Optimizer

# https://discourse.julialang.org/t/documenting-elements-of-a-struct/64769/2
"""
A `Spl` object.

# Fields
- `H`: an `RObject` generated by `splines::bs()`
- `β`: the coefficients for the B-spline.
"""
mutable struct Spl{T <: AbstractFloat}
    # H::Matrix{T}
    H::RObject{RealSxp}
    β::AbstractVector{T}
end

mutable struct MonotoneCS
    B::AbstractMatrix{Float64}
    rB::RObject
    β::AbstractVector{Float64}
    fitted::AbstractVector{Float64}
end

mutable struct MonotoneSS
    mx::Real
    rx::Real
    knots::AbstractVector{Float64}
    B::AbstractMatrix{Float64}
    Bend::AbstractMatrix{Float64}
    Bendd::AbstractMatrix{Float64}
    L::AbstractMatrix{Float64}
    β::AbstractVector{Float64}
    fitted::AbstractVector{Float64}
end

## define the negative of splines
Base.:-(x::MonotoneCS) = MonotoneCS(x.B, x.rB, -x.β, -x.fitted)
Base.:-(x::MonotoneSS) = MonotoneSS(x.mx, x.rx, x.knots, x.B, x.Bend, x.Bendd, x.L, -x.β, -x.fitted)

"""
    rcopy(s::Spl)

Convert RObject `s.H` as a Julia matrix, and `s.β` keeps the same.
"""
function rcopy(s::Spl)
    return rcopy(s.H), s.β
end

# deprecated? use mono_cs instead
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
    smooth_spline(x::AbstractVector, y::AbstractVector, xnew::AbstractVector)

Perform smoothing spline on `(x, y)`, and make predictions on `xnew`.

Returns: `yhat`, `ynewhat`,....
"""
function smooth_spline(x::AbstractVector{T}, y::AbstractVector{T}, xnew::AbstractVector{T}; keep_stuff = false, design_matrix = false) where T <: AbstractFloat
    spl = R"smooth.spline($x, $y, keep.stuff = $keep_stuff)"
    Σ = nothing
    if keep_stuff
        Σ = recover(rcopy(R"$spl$auxM$Sigma"))
    end
    B = nothing
    if design_matrix
        knots = rcopy(R"$spl$fit$knot")[4:end-3]
        bbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 4)"
        B = rcopy(R"predict($bbasis, $knots)")
    end
    λ = rcopy(R"$spl$lambda")
    coef = rcopy(R"$spl$fit$coef")
    return rcopy(R"predict($spl, $x)$y"), rcopy(R"predict($spl, $xnew)$y"), Σ, λ, spl, B, coef
end

"""
    predict(model::Spl{T}, xs::AbstractVector{T})
    predict(X::Vector{Float64}, y::Vector{Float64}, J::Int, Xnew::AbstractVector{Float64}, ynew::AbstractVector{Float64}
    predict(X::Vector{Float64}, y::Vector{Float64}, J::Int, Xnew::Vector{Float64}, ynew::Vector{Float64}, σ::Vector{Float64}

Make prediction based on fitted `Spl` on new points `xs`. If `Xnew` is provided, then also returns the prediction error `‖yhat - ynew‖_2^2`.
"""
function predict(model::Spl{T}, xs::AbstractVector{T}) where T <: AbstractFloat
    # the warning evaluated at outside point can be safely ignored since
    # the extrapolation of the function is the linear combinations of the extrapolation of each basis function
    # NB: it is different from smoothing splines, whose extrapolation is linear. (Clouds#85)
    return rcopy(R"suppressWarnings(predict($(model.H), $xs))") * model.β
end

function cv_err(x::AbstractVector{T}, y::AbstractVector{T}; nfold = 10, J = 10) where T <: AbstractFloat
    n = length(y)
    folds = div_into_folds(n, K = nfold)
    err = zeros(nfold)
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        spl = fit(x[train_idx], y[train_idx], J, "monotone", "increasing")
        yhat = predict(spl, x[test_idx])
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

function ci_mono_ss(x::AbstractVector, y::AbstractVector, λ = 1.0; ε = (eps())^(1/3), 
                                                                   prop_nknots = 1.0, B = 1000,
                                                                   α = 0.05)
    yhat = mono_ss(x, y, λ; ε = ε, prop_nknots = prop_nknots).fitted
    σhat = std(y - yhat)
    n = length(y)
    Yhat = zeros(n, B)
    for b = 1:B
        ystar = yhat + randn(n) * σhat
        Yhat[:, b] = mono_ss(x, ystar, λ; ε = ε, prop_nknots = prop_nknots).fitted
    end
    YCI = hcat([quantile(t, [α/2, 1-α/2]) for t in eachrow(Yhat)]...)'
    return yhat, YCI
end

"""
    mono_cs(x::AbstractVector, y::AbstractVector, J::Int = 4; increasing::Bool = true)

Monotone splines with cubic splines.
"""
function mono_cs(x::AbstractVector, y::AbstractVector, J::Int = 4; increasing::Bool = true)
    if !increasing
        return -mono_cs(x, -y, J, increasing = true)
    end
    B, rB = build_model(x, J)
    βhat, yhat = mono_ss(B, y, zeros(J, J), J)
    return MonotoneCS(B, rB, βhat, yhat)
end

"""
    mono_ss(x::AbstractVector, y::AbstractVector, λ = 1.0; prop_nknots = 1.0)

Monotone splines with smoothing splines, return a `MonotoneSS` object.
"""
function mono_ss(x::AbstractVector, y::AbstractVector, λ = 1.0; ε = (eps())^(1/3), prop_nknots = 1.0, increasing = true)
    if !increasing
        return -mono_ss(x, -y, λ; ε = ε, prop_nknots = prop_nknots, increasing = true)
    end
    B, L, J, mx, rx, idx, idx0, Bend, Bendd, knots = build_model(x, prop_nknots = prop_nknots)
    βhat, yhat = mono_ss(B, y, L, J, λ; ε = ε)
    return MonotoneSS(mx, rx, knots, B, Bend, Bendd, L, βhat, yhat)
end

function predict(W::MonotoneCS, xnew::AbstractVector)
    Bnew = rcopy(R"suppressWarnings(predict($(W.rB), $xnew))")
    return Bnew * W.β
end

function predict(W::MonotoneSS, xnew::AbstractVector)
    xnewbar = (xnew .- W.mx) ./ W.rx
    ind_right = xnewbar .> 1
    ind_left = xnewbar .< 0
    ind_middle = 0 .<= xnewbar .<= 1
    xm = xnewbar[ind_middle]
    n = length(xnew)
    if sum(ind_middle) == 0
        Bnew = zeros(0, length(W.β))
    else
        # xm cannot be empty
        Bnew = rcopy(R"splines::bs($xm, intercept = TRUE, knots=$(W.knots[2:end-1]))")
    end
    yhat = zeros(n)
    yhat[ind_middle] = Bnew * W.β
    boundaries = W.Bend * W.β
    slopes = W.Bendd * W.β
    yhat[ind_left] = boundaries[1] .+ slopes[1] * (xnewbar[ind_left] .- 0)
    yhat[ind_right] = boundaries[2] .+ slopes[2] * (xnewbar[ind_right] .- 1)
    return yhat
end

# shared B
"""
    mono_ss(B::AbstractMatrix, y::AbstractVector, L::AbstractMatrix, J::Int, λ::AbstractFloat)

Monotone Fitting with Smoothing Splines given design matrix `B` and cholesky-decomposed matrix `L`.

## Returns

- `βhat`: estimated coefficient
- `yhat`: fitted values
- (optional) `B` and `L`
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
function cv_mono_ss(x::AbstractVector{T}, y::AbstractVector{T}, λs = exp.(-6:0.5:1); ε = (eps())^(1/3), nfold = 10, increasing = true) where T <: AbstractFloat
    if !increasing
        return cv_mono_ss(x, -y, λs; ε = ε, nfold = nfold, increasing = true)
    end
    B, L, J = build_model(x)
    n = length(y)
    folds = div_into_folds(n, K = nfold, seed = -1)
    nλ = length(λs)
    err = zeros(nfold, nλ)
    for k = 1:nfold
        test_idx = folds[k]
        train_idx = setdiff(1:n, test_idx)
        for (i, λ) in enumerate(λs)
            βhat, _ = mono_ss(B[train_idx, :], y[train_idx], L, J, λ)
            if isnothing(βhat)
                @warn "βhat is estimated as nothing, so use average estimation for y"
                err[k, i] = norm(y[test_idx] .- mean(y[train_idx]))^2
            else
                err[k, i] = norm(B[test_idx, :] * βhat - y[test_idx])^2
            end
        end
    end
    return sum(err, dims = 1)[:] / n, B, L, J
end
