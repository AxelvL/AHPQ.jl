sqeuclidean(a,b) = sqrt(sum((a.-b).^2))
quantization_objective(sumBTB::AbstractMatrix, sumBTx::AbstractArray, C::AbstractArray) = dot((sumBTB*C),C)-(2*dot(sumBTx,C))