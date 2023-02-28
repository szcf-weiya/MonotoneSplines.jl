using Test
using MonotoneSplines

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
end

@testset "monotone splines with smoothing splines" begin
    yhat, yhatnew, _, λ = smooth_spline(x, y, x0)
    res = mono_ss(x, y, λ)
    y0hat = predict(res, x0)
    @test length(yhat) == length(res.fitted)
    @test length(y0hat) == length(y0) == length(yhatnew)
end

@testset "monotone fitting without smoothness penalty" begin
    err = MonotoneSplines.cv_err(x, y, nfold = 10, J = 10)
    @test 0 <= err < σ
end

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

@testset "confidence band width" begin
    @test MonotoneSplines.conf_band_width([0 1; 0 3]) ≈ 2.0
end

@testset "example functions" begin
    @test isa(cubic, Function)
    @test isa(logit, Function)
    @test isa(logit5, Function)
    @test isa(sinhalfpi, Function)
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