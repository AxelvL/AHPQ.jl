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
    for k in 1:qp.n_centers
        c_i = (j-1)*qp.n_centers*qp.n_dims_center + (k-1)*qp.n_dims_center + 1
        distances[k] = Distances.sqeuclidean(@view(data[d_i1:d_i2, i]), 
                                             @view(qp.C[c_i:(c_i+qp.n_dims_center-1)]))
    end
    choice = argmin(distances)
    local_loss = distances[choice]
    return local_loss, choice
end

"""
    function reset_assignments(qd::QuantizerData)
Helper function for L2 partition assignment.\n
Resets all cluster/codebook assignments in `qd.I`
"""

function reset_assignments!(qd::QuantizerData)
    Threads.@threads for i in 1:qd.n_dp
        qd.I.B[i] = spzeros(Bool, (qd.n_dims_center * qd.n_codebooks * qd.n_centers),qd.n_dim)
    end
    fill!(qd.I.assignments, 0)
end

"""
    function assignment_step(data::AbstractMatrix, qd::QuantizerData)
Assignment step for L2_loss: after having recomputed / initialized the codebooks,
we find for each u_j per datapoint x_i the favorable (l2-closest) cluster assignment.
"""
function assignment_step(data::AbstractMatrix, qd::QuantizerData)
    
    reset_assignments!(qd)
    total_loss = 0
    for i in 1:qd.n_dp
        for j in 1:qd.n_codebooks
            local_loss, c_ij = select_closest_center(data, qd, i, j)
            assign_cluster(qd, c_ij, i, j) # Saving selection
            total_loss += local_loss
        end     
    end
    return total_loss / qd.n_dp
end

