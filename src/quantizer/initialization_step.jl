struct Indexes{T<:AbstractVector}
    B::Vector{SparseMatrixCSC}
    assignments::Matrix{UInt16}
    IVF::T
end

struct QuantizerData{T<:AbstractVector}
    n_codebooks::Int
    n_centers::Int
    n_dp::Int
    n_dims::Int
    n_dims_center::Int
    I::Indexes
    C::T
end


"""
    function GenerateQuantizerData(data::AbstractMatrix; n_codebooks::Int, n_centers::Int)
Initialises an empty codebook / assignments data structure.
"""
function GenerateQuantizerData(data::AbstractMatrix, n_codebooks::Int, n_centers::Int)
    n_dim, n_dp = size(data)
    n_dims_center = n_dim ÷ n_codebooks
    C = zeros(n_dims_center*n_centers*n_codebooks)
    I = Indexes(Array{SparseMatrixCSC}(undef, n_dp), Matrix{UInt16}(undef, n_codebooks, n_dp),
                    [[Vector{Int}() for i in 1:n_centers] for j in 1:n_codebooks])
    return QuantizerData(n_codebooks, n_centers, n_dp, n_dim, n_dims_center, I, C)
end

function get_indexes(qd::QuantizerData, i::Int, j::Int)
    d_i1, d_i2 = (i-1)*qd.n_dims_center+1, i*qd.n_dims_center
    c_i = (i-1)*qd.n_centers*qd.n_dims_center + (j-1)*qd.n_dims_center + 1
    return d_i1, d_i2, c_i
end

function initialization!(data::AbstractMatrix, qd::QuantizerData, MT::MultiThreaded)
    Threads.@threads for i in 1:qd.n_codebooks
        random_indexes = shuffle!(collect(1:qd.n_dp))[1:qd.n_centers] # Pick random indexes
        for (j, idx) in enumerate(random_indexes)
            d_i1, d_i2, c_i = get_indexes(qd, i, j)
            qd.C[c_i:(c_i+qd.n_dims_center-1)] = deepcopy(data[d_i1:d_i2,idx])
        end
    end
end

function initialization!(data::AbstractMatrix, qd::QuantizerData, ST::SingleThreaded)
    for i in 1:qd.n_codebooks
        random_indexes = shuffle!(collect(1:qd.n_dp))[1:qd.n_centers] # Pick random indexes
        for (j, idx) in enumerate(random_indexes)
            d_i1, d_i2, c_i = get_indexes(qd, i, j)
            qd.C[c_i:(c_i+qd.n_dims_center-1)] = deepcopy(data[d_i1:d_i2,idx])
        end
    end
end

#####################################################################################
#                                                                                   #
#                                    Anisotropic                                    #
#                                                                                   #
#####################################################################################

struct AnisotropicWeights{F<:AbstractFloat}
    η::F
    h_orthog::F
    h_par::F
end

"""
    function ComputeWeightsFromT(n_dim::Int, T::AbstractFloat)
Computes η, orthogonal weight and parallel weight for the Anisotropic loss.
"""
function ComputeWeightsFromT(n_dim::Int, T::AbstractFloat)
    η = compute_η(T, n_dim)
    h_orthog = 1/(1+η)
    h_par    = 1 - h_orthog
    return AnisotropicWeights(η, h_orthog, h_par)
end