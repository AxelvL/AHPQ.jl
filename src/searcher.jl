"""
    function identify_relevant_dps(ahpq::AHPQdata, query::AbstractArray)

Computes the inner products between the query and the input pre-clustering vectors.
Returns the datapoints assigned to the closest `config[:b]` centers and the corresponding distances.
"""

function identify_relevant_dps(ahpq::AHPQdata, query::AbstractArray)
    n_centers = size(ahpq.clusterdata.centers)[2]
    distances = zeros(n_centers)
    for i in 1:n_centers
        @inbounds distances[i] = dot(query, @inbounds @view ahpq.clusterdata.centers[:,i])
    end
    cluster_indexes = partialsortperm(distances, 1:ahpq.config[:b], rev=true)
    dp_to_inspect = ahpq.clusterdata.assignments[cluster_indexes]
    return dp_to_inspect, distances[cluster_indexes]
end

struct cluster_iterator
    codebook_i::Int
    n_centers::Int
    n_dims_center::Int
end

function Base.iterate(f::cluster_iterator,stepid=0)
    # state the ending conditions first
    stepid += 1
    if f.n_centers < stepid
        return nothing
    end
    return ((stepid,(f.codebook_i-1)*f.n_centers*f.n_dims_center + (stepid-1)*f.n_dims_center + 1),stepid)
end


"""
    function create_complete_lookup_table(qd::QuantizerData, query::AbstractArray)

Helper function of the main `MIPS` function.\n
Computes the inner products between all codebook centers and the query. Distances are stored in a complete lookup table and is returned to speed up searching. 
"""
function create_complete_lookup_table(qd::QuantizerData, query::AbstractArray)
    lookup_table = zeros(qd.n_codebooks, qd.n_centers)
    for i in 1:qd.n_codebooks
        clust_iterator = cluster_iterator(i, qd.n_centers, qd.n_dims_center)
        @inbounds for (j, c_i) in clust_iterator
            @fastmath lookup_table[i,j] = dot(
                (@view(query[((i-1)*qd.n_dims_center+1):i*qd.n_dims_center])), 
                (@view(qd.C[c_i:(c_i+qd.n_dims_center-1)])))
        end
    end
    return lookup_table
end
"""
    function compute_distances(qd::QuantizerData, dp_ids::AbstractArray, LUT::AbstractMatrix, distances::Int, norms::AbstractArray)   

Computes the approximate inner products between the datapoints and the query, based on preclustering distance and the PQ distance.
Returns a vector of distances for all datapoints in `dp_ids`, using the lookup table `LUT` and preclustering distances `distances`.
"""
function compute_distances(qd::QuantizerData, dp_ids::AbstractArray, LUT::AbstractMatrix, distances::AbstractArray, norms::AbstractArray)
    distances_new = Float32[]
    dp_i = 0
    for (distance_i, bin) in enumerate(dp_ids)
        for dp in bin
            dp_i += 1
            push!(distances_new, distances[distance_i])
            j = UInt8(0)
            
            @inbounds for k in @inbounds @view qd.I.assignments[:,dp]
                j+= UInt8(1)
                distances_new[dp_i] += norms[dp]*LUT[j,k]
            end
        end            
    end
    distances_new
end

function compute_distances(qd::QuantizerData, dp_ids::AbstractArray, LUT::AbstractMatrix, distances::Int, norms::AbstractArray)
    n_dp = length(dp_ids)
    distances = zeros(n_dp)
    for (i,ii) in enumerate(dp_ids)
        j = UInt8(0)
        @inbounds for k in @inbounds @view qd.I.assignments[:,ii]
            j+= UInt8(1)
            @inbounds distances[i] += @inbounds LUT[j,k]
        end
    end
    return distances
end

"""
    function MIPS(ahpq::AHPQdata, query::AbstractArray, k::Int)

Searches the most `k` max inner products for the input query/queries for the
given indexes and configurations `ahpq`. Takes a single query-vector or a set of queries (matrix).

# Example
```julia
traindata = rand(d, n)
queries = rand(d, m)
ahpq = builder(traindata, 0.2)

k_neares_neighbours = MIPS(ahpq, queries, 10)
k_neares_neighbours = MIPS(ahpq, queries[:,1], 10)
"""
function MIPS(ahpq::AHPQdata, query::AbstractArray, k::Int)
    
     # Step 1
    if ahpq.config.a > 0
        dp_ids, distances = identify_relevant_dps(ahpq, query)
    else
        dp_ids = 1:ahpq.qd.n_dp
        distances = 0
    end

    # Step 2
    LUT = create_complete_lookup_table(ahpq.qd, query)
    distances = compute_distances(ahpq.qd, dp_ids, LUT, distances, ahpq.norms)

    # Step 3 -- Using reordering settings
    if ahpq.config.reorder == 0
        return @inbounds collect(Iterators.flatten(dp_ids))[partialsortperm(distances, 1:k, rev=true)]
    else
        dp_ids_searcher = partialsortperm(distances, 1:ahpq.config[:reorder], rev=true)
        dp_real_ids = collect(Iterators.flatten(dp_ids))
        exact_distances = dropdims(query'ahpq.residuals[:,dp_real_ids[dp_ids_searcher]],dims=1)
        distances[dp_ids_searcher] .+= exact_distances
        return @inbounds @view dp_real_ids[partialsortperm(distances, 1:k, rev=true)]
    end
end

function MIPS(ahpq::AHPQdata, queries::AbstractMatrix, n_neighbors::Int)
    n_queries = size(queries)[2]
    ranking = Array{Int64,2}(undef, n_neighbors, n_queries)
    Threads.@threads for j in 1:n_queries
        ranking[:,j] = MIPS(ahpq, @inbounds(@view(queries[:,j])), n_neighbors)
    end
    return ranking
end