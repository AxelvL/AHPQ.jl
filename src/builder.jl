struct AHPQdata
    clusterdata
    qd::QuantizerData
    residuals::AbstractMatrix
    norms::AbstractArray
    config::NamedTuple
end

"""
    function builder(data::Matrix; T::Real, kwargs...)

Main constructor for building the (Anisotropic) Hierarchical Product Quantizer.\n 
Leave out the keyword parameter (T::AbstractFloat) to train on Euclidean loss instead of Anisotropic loss
    
# Main Arguments
 * `Matrix` input data
 * `T::AbstractFloat` Threshold parameter of the Anisotropic loss function, set to 0 to use L2_loss

# Kwargs...
 * `T_preclustering::Int=-1` the `T` parameter for the preclustering loss function, set to -1 to use the same value as for `T`, 0 to use L2_loss
 * `n_codebooks::Int=n_dims/2` number of codebooks to train
 * `n_centers::Int=8` number of clusters/centers to train
 * `a::Int=√n_dp` number of clusters generated in preclustering step, set to 0 to turn off
 * `b::Int=(n_clusters_to_generate ÷ 5 + 2)` number of clusters pruned from preclustering step
 * `initialise_with_euclidean_loss::Bool=true` switches euclidean pre-training on/off
 * `inverted_index::Bool=true` switches to IVF methods for Euclidean quantization
 * `max_iter::Int=1000` the max number of iterations of the assignment_step-codebook_update loop
 * `stopcond<:AbstractFloat=1e-2` the stopcondition (l2-distance of codebook update) for the assignment_step-codebook_update loop 
 * `verbose::Bool=false` switches training information updates on/off
 * `max_iter_assignments::Int=10` maximum iterations for the approximate assignment step of anisotorpic quantization
 * `optimisation::String="exact"` the optimisation method for the codebook udpate step, takes: ("exact", "nesterov")
 * `multithreading::Bool=false` switches multi-threading on/off
 * `GPU::Bool=false` switches optimization on GPU on/off (uses CUDA.jl)
 * `increment_steps::Int=floor(log10(training_points))-2)` number of incremental steps, use 0 to switch off incremental training
 * `reorder::Int=0` number of datapoints used for exact inner product reordering
 * `training_points::Int=ceil(n_dp/5))` number of training points to train on

 # Example
 ```julia
 traindata = rand(d, n)
 l2_searcher          = builder(traindata; n_codebooks=50,
                                           n_centers=16)
 anisotropic_searcher = builder(traindata, 0.2; n_codebooks=50,
                                                n_centers=16)    
 ```
"""
function builder(data::Matrix; T::Real, kwargs...)
    # Step 0 - Retrieve configurations
    n_dims, n_dp = size(data)
    config, T = check_kwargs(kwargs, n_dp, n_dims, T)
    η = if T > 0 ComputeWeightsFromT(n_dims, T) else L2_loss() end
    η_cd = if (config.T_preclustering > 0) ComputeWeightsFromT(n_dims, config.T_preclustering) else L2_loss() end
    
    # Step 1 - Preclustering
    if config[:a] > 0
        if config[:verbose] @info("Generating precluster data...") end
        clusterdata =  GenerateClusterData(data, η_cd, config)

        if config[:verbose] @info("Computing residuals...") end
        for (i, center) in enumerate(eachcol(clusterdata.centers))
             data[:,clusterdata.assignments[i]] .-= center
        end
        norms = mapslices(norm, data, dims=1)
    else 
        clusterdata = nothing 
        norms = ones(n_dp)' 
    end

    # Step 2 - Quantization
    codebook=0
    traindata = if config.training_points > 0 subsample(config.training_points, data./norms) else data./norms end
    if config[:increment_steps] > 0
        if config[:verbose] @info("Starting incremental initialisation...") end
        qd = incremental_quantization(traindata, η, config) 
    else 
        if (config[:initialise_with_l2_loss]) & (T > 0)
            if config[:verbose] @info("Starting L2 initialisation...") end
            qd = quantizer(traindata, L2_loss(), codebook, merge(config,(;verbose=false))) 
            codebook = deepcopy(qd.C)
        end
        if config[:verbose] @info("Finished initialisation...") end
        qd = quantizer(traindata, η, codebook, config)
    end
    codebook = deepcopy(qd.C)
    if config.training_points > 0
        qd = GenerateQuantizerData(data, config.n_codebooks, config.n_centers)
        qd.C[:].= codebook
        thread = if config.multithreading MultiThreaded() else SingleThreaded() end
        assignment_step!(data, qd, BMatrix(), thread)
    end

    # Step 3 - Generate residuals for exact reordering
    for i in 1:n_dp
        data[:,i] .-= norms[i].*(qd.I.B[i]'qd.C)
    end

    return AHPQdata(clusterdata, qd, data, norms, config)
end