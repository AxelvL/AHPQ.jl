struct ClusterData{V<:AbstractVector,A<:AbstractArray}
    assignments::V
    centers::A
end

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