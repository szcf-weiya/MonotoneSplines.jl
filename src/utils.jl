using RCall
using StatsBase
using LinearAlgebra # for 1.0I

"""
    div_into_folds(N::Int; K = 10, seed = 1234)

Equally divide `1:N` into `K` folds with random seed `seed`. If `seed` is negative, it is a non-random division, where the `i`-th fold would be the `i`-th equidistant range.
"""
@inline function div_into_folds(N::Int; K::Int = 10, seed = 1234)
    if seed >= 0
        idxes = sample(MersenneTwister(seed), 1:N, N, replace = false)
    else
        idxes = collect(1:N)
    end
    # maximum quota per fold
    n = Int(ceil(N/K))
    # number folds for the maximum quota
    k = N - (n-1)*K
    # number fols for n-1 quota: K-k
    folds = Array{Array{Int, 1}, 1}(undef, K)
    for i = 1:k
        folds[i] = idxes[collect(n*(i-1)+1:n*i)]
    end
    for i = 1:K-k
        folds[k+i] = idxes[collect((n-1)*(i-1)+1:(n-1)*i) .+ n*k]
    end
    return folds
end

"""
    pick_knots(x::AbstractVector{T})

Partial code for picking knots in R's `smooth.spline`.

The source code of `smooth.spline` can be directly accessed via typing `smooth.spline` is an R session. Note that there might be different in different R versions. The code is adapted based on R 3.6.3.
"""
# refer to `smooth.spline`
function pick_knots(x::AbstractVector{T}; tol = 1e-6 * iqr(x), all_knots = false, scaled = false) where T <: AbstractFloat
    xx = round.(Int, (x .- mean(x)) / tol )
    # https://stackoverflow.com/questions/50899973/indices-of-unique-elements-of-vector-in-julia
    # unique index (Noted in techNotes)
    ud = unique(i -> xx[i], 1:length(xx))
    ux = sort(x[ud])
    idx0 = sortperm(x[ud])
    nx = length(ux)
    if all_knots
        nknots = nx
    else
        nknots = Int(rcopy(R".nknots.smspl($nx)"))
    end
    idx = round.(Int, range(1, nx, length = nknots))
    if scaled
        rx = (ux[end] - ux[1])
        mx = ux[1]
        ux = (ux .- mx) ./ rx
        return ux[idx], mx, rx, (1:length(x))[ud][idx0][idx], (1:length(x))[ud][idx0]
    else
        # to keep the same output
        return ux[idx], nothing, nothing, nothing, nothing 
    end
end

"""
    coverage_prob(CIs::AbstractMatrix, y0::AbstractVector)

Calculate coverage probability given `n x 2` CI matrix `CIs` and true vector `y0` of size `n`.
"""
function coverage_prob(CIs::AbstractMatrix, y0::AbstractVector)
    @assert size(CIs, 1) == length(y0)
    @assert size(CIs, 2) == 2
    return sum((CIs[:, 1] .<= y0) .* (CIs[:, 2] .>= y0)) / length(y0)
end

"""
    conf_band_width(CIs::AbstractMatrix)

Calculate width of confidence bands.
"""
function conf_band_width(CIs::AbstractMatrix)
    len = CIs[:, 2] - CIs[:, 1]
    return mean(len)
end

# cubic splines
"""
    build_model(x::AbstractVector{T}, J::Int)
    build_model(x::AbstractVector{T}, scaled::Bool)
    build_model(x::AbstractVector{T})

## Returns

- B-spline design matrix `B` at `x` for cubic splines
- `Bnew` at new points `xnew` if it is not `nothing`
- `L = nothing`: keep same return list for model of smoothing splines
- `J`: number of basis functions, which does not change for cubic splines, so it is only intended for smoothing splines 
- `mx, rx, idx, idx0`: the remaining only for smoothing splines
"""
function build_model(x::AbstractVector{T}, J::Int; xboundary = nothing, xnew = nothing, ε = (eps())^(1/3)) where T <: AbstractFloat
    Bnew = nothing
    if isnothing(xboundary)
        xboundary = [minimum(x), maximum(x)]
    end
    rB = R"splines::bs($x, df=$J, intercept=TRUE, Boundary.knots = $xboundary)" # needed for evaluating xnew
    B = rcopy(rB)
    if !isnothing(xnew)
        Bnew = rcopy(R"predict($rB, $xnew)")
    end
    L = nothing
    return B, Bnew, L, J
end

# smoothing splines
function build_model(x::AbstractVector{T}, scaled::Bool; xboundary = nothing, all_knots = false, xnew = nothing, ε = (eps())^(1/3)) where T <: AbstractFloat
    Bnew = nothing
    knots, mx, rx, idx, idx0 = pick_knots(ifelse(isnothing(xboundary), x, vcat(x, xboundary)), all_knots = all_knots, scaled = scaled)
    bbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 4)"
    Ω = rcopy(R"fda::eval.penalty($bbasis, 2)")
    # Ω = (Ω + Ω') / 2
    # correct the diagonal
    Ω += ε * 1.0I
    if scaled
        xbar = (x .- mx) ./ rx
        # see https://github.com/szcf-weiya/Clouds/issues/99#issuecomment-1272015742
        xbar[xbar .< 0] .= 0
        xbar[xbar .> 1] .= 1
        B = rcopy(R"fda::eval.basis($xbar, $bbasis)")
    else
        B = rcopy(R"fda::eval.basis($x, $bbasis)")
    end
    if !isnothing(xnew)
        if scaled
            xnewbar = (xnew .- mx) ./ rx
            # see https://github.com/szcf-weiya/Clouds/issues/99#issuecomment-1272015742
            xnewbar[xnewbar .< 0] .= 0
            xnewbar[xnewbar .> 1] .= 1
            # Bnew = rcopy(R"predict($bbasis, $xnewbar)")
            # force to extrapolation
            # Bnew = rcopy(R"splines::bs($xnewbar, intercept = TRUE, knots=$(knots[2:end-1]), Boundary.knots = $(knots[[1,end]]))")
            # automatically set boundary points as the range
            Bnew = rcopy(R"splines::bs($xnewbar, intercept = TRUE, knots=$(knots[2:end-1]))")
        else
            Bnew = rcopy(R"predict($bbasis, $xnew)") # no need to specify `fda::`
        end
    end
    J = length(knots) + 2
    L = nothing
    try
        L = Matrix(cholesky(Symmetric(Ω)).L)
    catch e
        @warn e
        ## perform pivoted Cholesky
        L = Matrix(cholesky(Symmetric(Ω), Val(true), check = false, tol = ε).L)
    end
    return B, Bnew, L, J, mx, rx, idx, idx0
end

function build_model(x::AbstractVector{T}; J = 10, xboundary = nothing, t = nothing, xnew = nothing, ε = (eps())^(1/3), all_knots = false, λ = nothing) where T <: AbstractFloat
    # t is the threshold, and λ is the corresponding Lagrange multipler
    if isnothing(t) & isnothing(λ)
        return build_model(x, J, xboundary = xboundary, xnew = xnew, ε = ε)
    else
        # to agree with old code, if t is given instead of lambda, scaled = false
        return build_model(x, !isnothing(λ),xboundary = xboundary, all_knots = all_knots, xnew = xnew, ε = ε)
    end
end