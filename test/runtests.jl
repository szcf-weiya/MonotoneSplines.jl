using Test
using MonotoneSplines

@testset "Jaccard Index" begin
    @test jaccard_index([0, 1], [0, 1]) == 1
    @test jaccard_index([0, 1.0], [0.5, 1.5]) == 1/3
    @test jaccard_index([0, 1], [-1, 1]) == 0.5
end

@testset "check confidence bands" begin
    check_CI(nrep = 1, nepoch0 = 1, nepoch = 1, fig = false, check_acc = false) #do not run on laptop, it might cause the PC dead
end
