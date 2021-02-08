sqeuclidean(a,b) = sqrt(sum((a.-b).^2))
quantization_objective(sumBTB::AbstractMatrix, sumBTx::AbstractArray, C::AbstractArray) = dot((sumBTB*C),C)-(2*dot(sumBTx,C))

"""
    function decompose_error(x_i, x_approx_i)
        
Computes the parallel and orthogonal error of the approximated codeword `̃x_i` w.r.t. the original data point `x_i`.
h_par = x((x - ̃x)⋅x) / (x/x⋅x).\n
h_orthog = x-̃x - h_par
"""
function decompose_error(x_i, x_approx_i)
    parallel = (dot((x_i - x_approx_i), x_i)*x_i)/dot(x_i,x_i)
    orthog = (x_i - x_approx_i) - parallel
    return parallel, orthog
end

"""
    function anisotropic_loss(x_i::AbstractArray, x_approx_i::AbstractArray, η::AnisotropicWeights)

Computes and returns the anisotropic loss between the approximated codeword `̃x_i` and 
the original data point `x_i`, based on the input weights for the parallel loss and orthogonal loss `η`.
"""
function anisotropic_loss(x_i::AbstractArray, x_approx_i::AbstractArray, η::AnisotropicWeights)
    parallel, orthog = decompose_error(x_i, x_approx_i)
    return η.h_par * dot(parallel,parallel) + η.h_orthog * dot(orthog,orthog)
end