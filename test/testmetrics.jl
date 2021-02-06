using AHPQ
using Test

@testset "testmetrics" begin
    # Data Generation
    a = Array{Int}(undef, 10, 4)
    for i in 1:4 a[:,i] = collect(1:10) end
    b = deepcopy(a)

    @info("Testing RecallFull")
    @test recallfull(a, b)==1.0
    b[9,1] = 12

    recallfull(a,b)
    @test recallfull(a, b)==0.975

    @info("Testing RecallN")
    @test recallN(a, b, 10) == 0.975
    @test recallN(a, b, 5) == 1.0

    @info("Testing Recall1atN")
    c = Array{Int}(undef, 10, 4)
    for i in 1:4 c[:,i] = collect(10:19) end
    c[5,1] = 1
    @test recall1atN(c, a, 10) == 0.25
    @test recall1atN(c, a, 4) == 0
    @test recall1atN(b, a, 1) == 1

    @test get1atNscores(c, a, 5) == [0.0, 0.0, 0.0, 0.0, 0.25]

    @info("Testing Approx_error")

    # Data prep
    n_dp = 100
    n_dim = 4
    traindata = rand(n_dim, n_dp)
    queries = rand(n_dim, 10)
    innerproducts = traindata' * queries
    n_neighbors=10
    rtrue = mapslices(x -> partialsortperm(x, 1:n_neighbors, rev=true), innerproducts,dims=1);

    # Train AHPQ
    ahpq = builder(deepcopy(traindata); T=0, n_codebooks=1, n_centers=2, training_points=0)
    approx_error(ahpq.qd, deepcopy(traindata), rtrue, queries)

    @test true
end