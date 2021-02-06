using AHPQ
using Test

@testset "searcher" begin
    using Statistics: norm
    n_dp = 1000
    n_dim = 4
    data = rand(n_dim, n_dp)
    data = data ./ mapslices(norm, data, dims=1)
    queries = rand(4, 100)

    stopcond = 1e-1
    @info("Searcher without preclustering and without reorder")
    ahpq = builder(deepcopy(data), T=0.6, a=0, reorder=0)
    MIPS(ahpq, queries, 10)
    MIPS(ahpq, queries[:,1], 10)
    @test true
    
    @info("Searcher with preclustering and with reorder")
    ahpq = builder(deepcopy(data), T=0.6, a=10, reorder=25)
    MIPS(ahpq, queries, 10)
    MIPS(ahpq, queries[:,1], 10)
    @test true

    @info("Searcher without preclustering and with reorder")
    ahpq = builder(deepcopy(data), T=0.6, a=0, reorder=25)
    MIPS(ahpq, queries, 10)
    MIPS(ahpq, queries[:,1], 10)
    @test true

    @info("Searcher with preclustering and without reorder")
    ahpq = builder(deepcopy(data), T=0.6, a=10, reorder=0)
    MIPS(ahpq, queries, 10)
    MIPS(ahpq, queries[:,1], 10)
    @test true
end