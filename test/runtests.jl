using Test
using MonotoneSplines
using RCall
__init_pytorch__()
TEST_MLP = true # use locally, since the time cost is relatively hight

@testset "Jaccard Index" begin
    @test jaccard_index([0, 1], [0, 1]) ≈ 1
    @test jaccard_index([0, 1.0], [0.5, 1.5]) ≈ 1/3
    @test jaccard_index([0, 1], [-1, 1]) ≈ 0.5
    @test jaccard_index([0, 1], [0.2, 0.8]) ≈ 0.6 # 0.6000000000000001 == 0.6 Failed
end

@testset "optimization solution for monotone splines" begin
    n = 20
    x = rand(n)
    y = x .^2 + randn(n) * 0.1
    model = MonotoneSplines.bspline(x, y, 10)
    Ω = MonotoneSplines.eval_penalty(model, x)
    @test size(Ω) == (10, 10)
    λs = exp.(-6:0)
    res = MonotoneSplines.cv_mono_ss(x, y, λs)
    @test length(λs) == length(res[1])
end

n = 20
σ = 0.1
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, exp, seed = 1234);

@testset "monotone splines with cubic splines" begin
    res = mono_cs(x, y, 4)
    spl = MonotoneSplines.monotone_spline(x, y, 4)
    H, βhat2 = rcopy(spl)
    @test res.β ≈ βhat2 atol=1e-3
    @test res.B ≈ H atol=1e-3
    y0hat = predict(res, x0)
    @test length(y0hat) == length(y0)
    # decreasing
    res2 = mono_cs(x, -y, 4, increasing = false)
    @test res2.β ≈ -res.β atol = 1e-3
end

@testset "monotone splines with smoothing splines" begin
    yhat, yhatnew, _, λ = smooth_spline(x, y, x0)
    res = mono_ss(x, y, λ)
    y0hat = predict(res, x0)
    @test length(yhat) == length(res.fitted)
    @test length(y0hat) == length(y0) == length(yhatnew)
    # decreasing
    res2 = mono_ss(x, -y, λ, increasing = false)
    @test res2.β ≈ -res.β atol = 1e-3
end

@testset "monotone fitting without smoothness penalty" begin
    err = MonotoneSplines.cv_err(x, y, nfold = 10, J = 10)
    @test 0 <= err < σ
end

if TEST_MLP
    @testset "compare monotone fitting" begin
        λ = 1e-5;
        Ghat, loss = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, disable_progressbar = true);
        Ghat2, loss2 = mono_ss_mlp(x, y, λl = λ, λu = λ, device = :cpu, backend = "pytorch", disable_progressbar = true);
        @test isa(Ghat, Function)
        @test isa(Ghat2, Function)
        yhat = Ghat(y, λ);
        yhat2 = Ghat2(y, λ);
        yhat0 = mono_ss(x, y, λ, prop_nknots = 0.2).fitted;
        @test sum((yhat - yhat0).^2) / n < 1e-3
        @test sum((yhat2 - yhat0).^2) / n < 1e-3
    end

    @testset "compare confidence band between OPT solution and MLP generator" begin
        # adapted from examples/monoci_mlp.jl
        λl = 1e-2
        λu = 1e-1
        λs = range(λl, λu, length = 2)
        RES0 = [ci_mono_ss(x, y, λ, prop_nknots = 0.2) for λ in λs]
        Yhat0 = hcat([RES0[i][1] for i=1:2]...)
        YCIs0 = [RES0[i][2] for i = 1:2]
        Yhat, YCIs, LOSS = ci_mono_ss_mlp(x, y, λs, prop_nknots = 0.2, device = :cpu, backend = "flux", nepoch0 = 1, nepoch = 1, disable_progressbar = true);
        Yhat2, YCIs2, LOSS2 = ci_mono_ss_mlp(x, y, λs, prop_nknots = 0.2, device = :cpu, backend = "pytorch", nepoch0 = 1, nepoch = 1, disable_progressbar = true);
        overlap = [jaccard_index(YCIs[i], YCIs0[i]) for i = 1:2]
        overlap2 = [jaccard_index(YCIs2[i], YCIs0[i]) for i = 1:2]
        @test all(0.0 .< overlap .< 1.0)
        @test all(0.0 .< overlap2 .< 1.0)
    end

    @testset "check confidence bands" begin
        res1 = check_CI(nrep = 1, nepoch0 = 1, nepoch = 1, fig = false, check_acc = false, nB = 10, backend = "pytorch", prop_nknots = 0.2, nhidden = 100, niter_per_epoch = 2) 
        res2 = check_CI(nrep = 1, fig = false, check_acc = false, nB = 10, model_file = res1[end], prop_nknots = 0.2) 
        @test res1[3] ≈ res2[3] # res_err is not random like CI results
        res3 = check_CI(nrep = 1, nepoch0 = 1, nepoch = 1, fig = false, check_acc = false, nB = 10, backend = "flux", prop_nknots = 0.2, nhidden = 100, niter_per_epoch = 2, gpu_id = -1) 
        # with flux backend, both `**.bson` and `**_ci.bson` are saved, so if check_acc=false, use `_ci.bson`
        res4 = check_CI(nrep = 1, fig = false, check_acc = false, nB = 10, model_file = res3[end][1:end-5] *"_ci.bson", prop_nknots = 0.2, gpu_id = -1) 
        @test res3[3] ≈ res4[3]
    end
end

@testset "confidence band width" begin
    @test MonotoneSplines.conf_band_width([0 1; 0 3]) ≈ 2.0
end

R"""
recover.Sigma <- function(Sigma) {
    n = length(Sigma)
    # just 4p = n, NOT p + p-1 + p-2 + p-3 = 4p - 6 = n
    p = n/4
    res = matrix(0, nrow = p, ncol = p)
    diag(res) = Sigma[1:p]
    diag(res[1:(p-1),2:p]) = Sigma[(p+1):(2*p-1)]
    diag(res[2:p, 1:(p-1)]) = Sigma[(p+1):(2*p-1)]
    diag(res[1:(p-2),3:p]) = Sigma[(2*p+1):(3*p-2)]
    diag(res[3:p, 1:(p-2)]) = Sigma[(2*p+1):(3*p-2)]
    diag(res[1:(p-3),4:p]) = Sigma[(3*p+1):(4*p-3)]
    diag(res[4:p, 1:(p-3)]) = Sigma[(3*p+1):(4*p-3)]
    return(res)
  }
"""
@testset "smooth.spline vs fda::" begin
    x = rand(100)
    y = x .^2 + randn(100) * 0.1
    sfit = R"smooth.spline($x, $y, keep.stuff = TRUE, all.knots = TRUE)"
    Σ = rcopy(R"recover.Sigma($sfit$auxM$Sigma)")
    XWX = rcopy(R"recover.Sigma($sfit$auxM$XWX)")
    λ = rcopy(R"$sfit$lambda")
    knots, xmin, rx, idx, idx0 = MonotoneSplines.pick_knots(x, all_knots = true)
    xbar = (x .- xmin) / rx
    @test xbar[idx] == knots
    @test rx == rcopy(R"$sfit$fit$range")
    @test xmin == rcopy(R"$sfit$fit$min")
    bbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 4)"
    Ω = rcopy(R"fda::eval.penalty($bbasis, 2)")
    B = rcopy(R"fda::eval.basis($knots, $bbasis)")
    n, p = size(B)
    @test sum((Ω - Σ).^2) / sum(Ω.^2) < 1e-3 # the absolute difference might be slightly larger, so normalize it by Ω
    @test XWX ≈ B' * B
    # only on unique values
    βhat = inv(B' * B + λ*Ω) * B' * y[idx]
    βhat1 = inv(B' * B + λ*Σ) * B' * y[idx]
    βhat0 = rcopy(R"$sfit$fit$coef")
    @test sum((βhat - βhat0).^2) / p < 1e-7
    @test sum((βhat1 - βhat0).^2) / p < 1e-7
end

@testset "check derivative of Bspline" begin
    n = 20
    x = rand(n)
    knots = 0:0.1:1.0 # keep same gap between knots
    J = length(knots) + 2
    extended_knots = vcat(zeros(3), knots, ones(3))
    # ξ(j+m-1) - ξ(j) # NB: the interval is not a constant
    δξ = extended_knots[2+3:J+3] - extended_knots[2:J]
    bbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 4)" 
    dbbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 3)" 
    B = rcopy(R"fda::eval.basis($x, $bbasis)")
    β = randn(size(B, 2))
    dB = rcopy(R"fda::eval.basis($x, $bbasis, Lfdobj=1)")
    dB2 = rcopy(R"fda::eval.basis($x, $dbbasis)")
    d1 = dB * β
    d2 = dB2 * ((β[2:end] - β[1:end-1]) ./ δξ) * 3
    @assert d1 ≈ d2
end

@testset "check boundary evaluation of Bspline" begin
    knots = 0:0.1:1.0
    J = length(knots) + 2
    extended_knots = vcat(zeros(3), knots, ones(3))
    # cubic
    bbasis = R"fda::create.bspline.basis(breaks = $knots, norder = 4)" 
    B = rcopy(R"fda::eval.basis($knots, $bbasis)")
    for i in [4, 5, 6, J-1, J, J+1]
        @test B[i-3, i-3:i-1] ≈ bs4_τi(extended_knots, i)
    end
    # @test B[1, 1:3] ≈ bs4_τi(extended_knots, 4) #τ_4 

    # quadratic
    bbasis2 = R"fda::create.bspline.basis(breaks = $knots, norder = 3)" 
    B2 = rcopy(R"fda::eval.basis($knots, $bbasis2)")
    for i in [4, 5, 6, J-1, J, J+1]
        @test B2[i-3, i-3:i-2] ≈ bs3_τi(extended_knots, i)
    end

    dB2 = rcopy(R"fda::eval.basis($knots, $bbasis, Lfdobj=2)")
    @test dB2[1, 1:3] ≈ 6/0.1 * [1 /0.1, -1/0.2-1/0.1, 1/0.2] # f''(τ_4)
end

@testset "check solution" begin
    x = 0:0.1:1.0
    y = x.^3
    res = mono_cs(x, y, 5) # borrow B
    β0 = [0.1, 0.2, 0.2, 0.4, 0.5]
    y0 = res.B * β0 + randn(length(x)) * 0.01 #NB: need to make sure res0.β indeed have β2 = β3, so larger noise might let the solution change dramatically
    res0 = mono_cs(x, y0, 5)
    G = zeros(4, 5) # γ1 < γ2 = γ3 < γ4 < γ5
    G[1, 1] = 1
    G[2, 2] = G[2, 3] = 1
    G[3, 4] = G[4, 5] = 1
    β1 = G' * inv(G * res.B' * res.B * G') * G * res.B' * y0
    if abs(res0.β[3] - res0.β[2]) < cbrt(eps()) # make sure the active set holds
        @assert sum((β1 - res0.β).^2) < sqrt(eps())
    end
end

@testset "check conditions for monotonicity" begin
    γ = [1, 2, 3, 4]
    ξ = [0, 1]
    τ = vcat(zeros(3), ξ, ones(3))
    bbasis = R"fda::create.bspline.basis(breaks = $ξ, norder = 4)"
    @test is_sufficient(γ)
    @test is_necessary(γ, τ, bbasis)
    @test is_sufficient_and_necessary(γ, τ, bbasis)
end

@testset "check conditions by areas" begin
    ## J = 4
    ξ = [0, 1]
    τ = vcat(zeros(3), ξ, ones(3))
    bbasis = R"fda::create.bspline.basis(breaks = $ξ, norder = 4)"
    γ1s = range(-10, 10, step = 1)
    γ2s = range(-10, 10, step = 1)
    res = zeros(length(γ1s), length(γ2s), 3)
    for (i, γ1) in enumerate(γ1s)
        for (j, γ2) in enumerate(γ2s)
            γ = [γ1, γ2, 3, 4]
            res[i, j, 1] = is_sufficient(γ)
            res[i, j, 2] = is_necessary(γ, τ, bbasis)
            res[i, j, 3] = is_sufficient_and_necessary(γ, τ, bbasis)
        end
    end
    areas = sum(res, dims = (1, 2))[:]
    @test areas[1] <= areas[3] <= areas[2]
end