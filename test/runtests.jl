using Test
using MonotoneSplines

@testset "Jaccard Index" begin
    @test jaccard_index([0, 1], [0, 1]) == 1
    @test jaccard_index([0, 1.0], [0.5, 1.5]) == 1/3
    @test jaccard_index([0, 1], [-1, 1]) == 0.5
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

@testset "check confidence bands" begin
    check_CI(nrep = 1, nepoch0 = 1, nepoch = 1, fig = false, check_acc = false, nB = 10) 
end
