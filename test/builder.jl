using AHPQ
using Test

@testset "builder" begin
    using Statistics: norm

    n_dp = 1000
    n_dim = 4
    data = rand(n_dim, n_dp)
    data = data ./ mapslices(norm, data, dims=1)

    stopcond = 1e-1

    @info("\nTesting preclustering")
    ahpq = builder(deepcopy(data); T=0.2, a=10, increment_steps = 0, verbose=true, stopcond=stopcond);
    ahpq = builder(deepcopy(data); T=0, a=10, increment_steps = 0, verbose=true, stopcond=stopcond);
    @test true

    @info("\nTesting Using different loss functions for preclustering")
    ahpq = builder(deepcopy(data); T=0.2, a=10, increment_steps = 0, verbose=true, stopcond=stopcond, T_preclustering=0.3); # Anisotropic, T=0.3
    ahpq = builder(deepcopy(data); T=0.2, a=10, increment_steps = 0, verbose=true, stopcond=stopcond, T_preclustering=-1);  # Anisotropic, T=T
    ahpq = builder(deepcopy(data); T=0.2, a=10, increment_steps = 0, verbose=true, stopcond=stopcond, T_preclustering=0);   # L2
    @test true

    @info("\nTesting L2 initialisation and incremental initialisation")
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 2, training_points=0, verbose=true, stopcond=stopcond);
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, initialise_with_l2_loss=true, verbose=true, stopcond=stopcond);
    @test true

    @info("\nTesting Inverted Index ")
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, inverted_index=true);
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, inverted_index=true);
    @test true

    @info("\nTesting different optimisation methods")
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="exact");
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="nesterov");
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="exact");
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="nesterov");
    @test true

    @info("\nTesting multithreading")
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="exact", multithreading=true);
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="nesterov", multithreading=true);
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="exact", multithreading=true);
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="nesterov", multithreading=true);
    @test true

    @info("\nTesting GPU")
    ahpq = builder(deepcopy(data); T=0, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="Nesterov", GPU=true);
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="Nesterov", GPU=true);
    @test true

    @info("\nTesting reorder function")
    ahpq = builder(deepcopy(data); T=0.2, a=0, increment_steps = 0, verbose=true, stopcond=stopcond, optimisation="nesterov", reorder=250);
    @test true
end
