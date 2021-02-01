#################################################################################################
#                                                                                               #
#                              L2 loss - Exact Assignment                                       #
#                                                                                               #
#################################################################################################

"""
    function select_closest_center(data::AbstractMatrix, qd::QuantizerData, i::Int, j::Int)    
Helper function for L2 partition assignment.\n
Selects the closest center for the given datapoint `i` and codebook index `j`
"""
function select_closest_center(data::AbstractMatrix, qd::QuantizerData, i::Int, j::Int)
    d_i1, d_i2 = (j-1)*qd.n_dims_center+1, j*qd.n_dims_center        
    distances = zeros(qd.n_centers)
    for k in 1:qd.n_centers
        c_i = (j-1)*qd.n_centers*qd.n_dims_center + (k-1)*qd.n_dims_center + 1
        distances[k] = sqeuclidean(@view(data[d_i1:d_i2, i]), 
                                   @view(qd.C[c_i:(c_i+qd.n_dims_center-1)]))
    end
    choice = argmin(distances)
    local_loss = distances[choice]
    return local_loss, choice
end


"""
    function reset_assignments(qd::QuantizerData, method::UpdateMethod)
Helper function for L2 partition assignment.\n
Resets all cluster/codebook assignments in `qd.I`
"""
function reset_assignments!(qd::QuantizerData, method::BMatrix)
    Threads.@threads for i in 1:qd.n_dp
        qd.I.B[i] = spzeros(Bool, (qd.n_dims_center * qd.n_codebooks * qd.n_centers),qd.n_dims)
    end
    fill!(qd.I.assignments, 0)
end

function reset_assignments!(qd::QuantizerData, method::InvertedIndex)
    qd.I.IVF[:] = [[Vector{Int}() for i in 1:qd.n_centers] for j in 1:qd.n_codebooks]
    fill!(qd.I.assignments, 0)
end


"""
    function assign_cluster(qd::QuantizerData, c_ij::Int, i::Int, j::Int, method::UpdateMethod)
Helper function for L2 partition assignment.\n 
Updates the indexes in `qd.B` for the selected method.
"""
function assign_cluster!(qd::QuantizerData, c_ij::Int, i::Int, j::Int, method::BMatrix, threading::Threading)
    b_1, b_2 = (j-1)*qd.n_dims_center+1, ((qd.n_centers * (j-1) * qd.n_dims_center) + 1) + ((c_ij-1)*qd.n_dims_center)
    qd.I.B[i][b_2: (b_2+qd.n_dims_center-1),b_1:(b_1+qd.n_dims_center-1)] = sparse(I, qd.n_dims_center, qd.n_dims_center)
    qd.I.assignments[j,i] = c_ij
end
function assign_cluster!(qd::QuantizerData, c_ij::Int, i::Int, j::Int, method::InvertedIndex, threading::SingleThreaded)
    append!(qd.I.IVF[j][c_ij],i)
    qd.I.assignments[j,i] = c_ij
end


"""
    function append_with_datalock(qd::QuantizerData,data_locks::Array{Bool, 2}, i::Int, j::Int, c_ij::Int)    
Helper function for the assignment_loop! function, specifically for the IVF MultiThreaded configuration to 
avoid data race while appending.
"""
function append_with_datalock(qd::QuantizerData,data_locks::Array{Bool, 2}, i::Int, j::Int, c_ij::Int)
    task_completed = false
    while !task_completed
        if data_locks[j,c_ij]
            @inbounds data_locks[j,c_ij] = false
            @inbounds append!(qd.I.IVF[j][c_ij],i)
            @inbounds data_locks[j,c_ij] = true
            task_completed=true
        end
    end
end

"""
    function assignment_loop!(data::AbstractMatrix, qd::QuantizerData, method::UpdateMethod, threading::Threading)
Helper function for L2 partition assingment.\n
Loops through all datapoints and all codebooks and assigns the subdimensions to the L2 closest center.
"""
function assignment_loop!(data::AbstractMatrix, qd::QuantizerData, method::BMatrix, threading::MultiThreaded)
    Threads.@threads for i in 1:qd.n_dp
        for j in 1:qd.n_codebooks
            local_loss, c_ij = select_closest_center(data, qd, i, j)
            assign_cluster!(qd, c_ij, i, j, method, threading)
        end
    end
    return nothing
end

function assignment_loop!(data::AbstractMatrix, qd::QuantizerData, method::InvertedIndex, threading::MultiThreaded)
    println("Config: InvertedIndex, MultiThreaded")
    data_locks = Array{Bool, 2}(undef, qd.n_codebooks, qd.n_centers)
    fill!(data_locks, true)
    Threads.@threads for i in 1:qd.n_dp
        for j in 1:qd.n_codebooks
            _, c_ij = select_closest_center(data, qd, i, j)
            append_with_datalock(qd, data_locks, i, j, c_ij)
            qd.I.assignments[j,i] = c_ij
        end
    end
    return nothing
end

function assignment_loop!(data::AbstractMatrix, qd::QuantizerData, method::UpdateMethod, threading::SingleThreaded)
    total_loss = 0
    for i in 1:qd.n_dp
        for j in 1:qd.n_codebooks
            local_loss, c_ij = select_closest_center(data, qd, i, j)
            assign_cluster!(qd, c_ij, i, j, method, threading)
            total_loss += local_loss
        end
    end
    return total_loss / qd.n_dp
end


"""
    function assignment_step(data::AbstractMatrix, qd::QuantizerData)
Assignment step for L2_loss: after having recomputed / initialized the codebooks,
we find for each u_j per datapoint x_i the favorable (l2-closest) cluster assignment.
Possible configuations: MultiThreading on/off, InvertedIndex on/off.
"""
function assignment_step!(data::AbstractMatrix, qd::QuantizerData, method::UpdateMethod, threading::Threading)
    reset_assignments!(qd, method)
    loss = assignment_loop!(data, qd, method, threading)
    return loss
end