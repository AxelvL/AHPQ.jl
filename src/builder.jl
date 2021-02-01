"""
    builder(data::Matrix, T::AbstractFloat; n_codebooks::Int, n_centers::Int, kwargs...)

Main constructor for building the (Anisotropic) Hierarchical Product Quantizer.\n 
Leave out the keyword parameter (T::AbstractFloat) to train on Euclidean loss instead of Anisotropic loss
    
# Main Arguments
 * `Matrix` input data

# Keyword Arguments
 * `T::AbstractFloat` Threshold parameter of the Anisotropic loss function, set to 0 to use L2_loss

# Optional Arguments
 * `n_codebooks::Int=n_dims/2` number of codebooks to train
 * `n_centers::Int=8` number of clusters/centers to train
 * `a::Int=√n_dp` number of clusters generated in preclustering step, set to 0 to turn off
 * `b::Int=(n_clusters_to_generate ÷ 5 + 2)` number of clusters pruned from preclustering step
 * `initialize_with_euclidean_loss::Bool=true` switches euclidean pre-training on/off
 * `inverted_index::Bool=true` switches to IVF methods for Euclidean quantization
 * `max_iter::Int=1000` the max number of iterations of the assignment_step-codebook_update loop
 * `stopcond<:AbstractFloat=9e-2` the stopcondition (l2-distance of codebook update) for the assignment_step-codebook_update loop 
 * `verbose::Bool=false` switches training information updates on/off
 * `max_iter_assignments::Int=10` maximum iterations for the approximate assignment step of anisotorpic quantization
 * `optimization::String="exact"` the optimization method for the l2_quantizer, takes: ("exact", "nesterov")
 * `multithreading::Bool=false` switches multi-threading on/off
 * `GPU::Bool=false` switches optimization on GPU on/off (requires CUDA.jl)
 * `increment_steps::Int=4` number of incremental steps, use 0 to switch off incremental training
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
    config = generate_data_dependent_defaults(n_dp, n_dims)
    config = merge(config, kwargs)
    config = check_kwargs(config, n_dims)
    return config
    # # Step 1 - Preclustering
    # if config[:precluster_data]
    #     # Step 1a - Generate Clusters
    #     if config[:verbose]
    #         @info("Generating precluster data...")
    #     end
    #     clusterdata =  GenerateClusterData(data, merge(config, (; use_inverted_index=true,
    #                                                               verbose=false)))

    #     # Step 1b - Generate residuals
    #     for (i, center) in enumerate(eachcol(clusterdata.centers))
    #         data[:,clusterdata.assignments[i]] .-= center
    #     end
    # else
    #     clusterdata = nothing
    # end

    # # Step 2 - Quantization
    # pqdata = GenerateQuantizerData(data; n_codebooks=n_codebooks, n_centers=n_centers)
    # if config[:incremental_fitting]
    #     pqdata = incremental_quantization(pqdata, config)
    # else
    #     pqdata = quantization(pqdata, config)
    # end

    # # Step 3 - Generate residuals for exact reordering
    # for i in 1:n_dp
    #     pqdata.data[:,i] .-= pqdata.encoding.B[i]'pqdata.encoding.C
    # end

    # return (preclusterdata = clusterdata, pqdata = pqdata, config = config)
end