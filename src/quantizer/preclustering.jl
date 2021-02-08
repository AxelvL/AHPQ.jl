struct ClusterData{V<:AbstractVector,A<:AbstractArray}
    assignments::V
    centers::A
end

"""
    function GenerateClusterData(data::AbstractMatrix, η::Weights, config::NamedTuple)   
         
Performs the preclustering step of the AHPQ; given a number of clusters `config[:a]` and performs Vector Quantization
using the main quantization function, using `n_codebooks=1` `n_centers=config[:a]` using the input loss function `η`. 
Can be set to L2_loss. The loss function is configured with the `T_preclustering` configuration before this function. 
"""
function GenerateClusterData(data::AbstractMatrix, η::Weights, config::NamedTuple)
    cd = quantizer(data, η, 0, merge(config, (; n_codebooks=1, n_centers=config[:a],
                                            verbose=false)))
    codebook = deepcopy(cd.C)
    cd = GenerateQuantizerData(data, 1, config.a)
    cd.C .= codebook
    assignment_step!(data, cd, InvertedIndex(), SingleThreaded())
    centers = reshape(cd.C, (size(data)[1], config[:a]))
    return ClusterData(cd.I.IVF[1], centers)
end