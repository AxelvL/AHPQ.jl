#################################################################################################
#                                                                                               #
#                                           L2 Loss                                             #
#                                                                                               #
#################################################################################################

##################################### B-Matrix methods ##########################################

"""
    function inner_loop_BTB!(qd::QuantizerData, diag::AbstractArray, Bsum, i)    

Inner loop of the BTB computation, finds the values in the diagonal by taking out the non-zero
values from ∑B.
"""
function inner_loop_BTB!(qd::QuantizerData, diag::AbstractArray, Bsum, i)
    i1_nzval = (i-1)*qd.n_dims_center*qd.n_centers+1
    i2_nzval = i1_nzval+qd.n_centers-1
    i1_diag = (i-1)*qd.n_centers+1
    i2_diag = i1_diag + qd.n_centers-1
    @inbounds diag[i1_diag:i2_diag] = @inbounds @view Bsum.nzval[i1_nzval:i2_nzval]
end

"""
    function compute_BTB(qd::QuantizerData, processing::MultiThreaded)    

Helper function of the codebook update step. Computes the ∑BTB matrix by summing the sparse matrices
and deriving the values on the diagonal from the non-zero values of the result to speed up the computaiton.
"""
function compute_BTB(qd::QuantizerData, processing::SingleThreaded)
    Bsum = sum(qd.I.B)
    diag = zeros(Int, qd.n_codebooks*qd.n_centers)
    for i in 1:qd.n_codebooks
        inner_loop_BTB!(qd, diag, Bsum, i)
    end
    return Diagonal(repeat(diag, inner=qd.n_dims_center))
end

function compute_BTB(qd::QuantizerData, processing::MultiThreaded)
    Bsum = sum(qd.I.B)
    diag = zeros(Int, qd.n_codebooks*qd.n_centers)
    Threads.@threads for i in 1:qd.n_codebooks
        inner_loop_BTB!(qd, diag, Bsum, i)
    end
    return Diagonal(repeat(diag, inner=qd.n_dims_center))
end

"""
    function compute_BTx(data::AbstractMatrix, qd::QuantizerData)    

Computes the ∑BTx matrix from the quantizer and the input data.
"""
function compute_BTx(data::AbstractMatrix, qd::QuantizerData)
    Bsize = (qd.n_dims_center * qd.n_codebooks * qd.n_centers)
    sumBTx = zeros(Bsize)
    for i in 1:qd.n_dp
        sumBTx+= qd.I.B[i]*@view(data[:,i])
    end
    return sumBTx
end

function compute_BMatrices(data::AbstractMatrix, qd::QuantizerData, processing::Processing)
    BTB = compute_BTB(qd, processing)
    BTx = compute_BTx(data, qd)
    return BTB, BTx
end

"""
    function compute_BMatrices(data::AbstractMatrix, qd::QuantizerData, processing::GPU)

Main helper function of the codebook update step. Calls the right functions to compute the BTB and
BTx matrix from the data and configurations. Returns them as a normal array or `CuArray` depending
on the processing setting.
"""
function compute_BMatrices(data::AbstractMatrix, qd::QuantizerData, processing::GPU)
    BTB = compute_BTB(qd, MultiThreaded())
    BTx = compute_BTx(data, qd)
    BTB = CuArray(BTB)
    BTx = CuArray(BTx)
    return BTB, BTx
end

"""
function optimisation!(BTB::AbstractMatrix, BTx::AbstractArray, update_method::UpdateMethod, 
    optimisation_method::OptimizationMethod, processing::Processing)

Computes the new codebook given the BTB and BTx marix and the optimisation method. Allows
for Exact updating (BTB^-1 ⋅ BTx) and Approximate (Nesterov) updating using ∇(BTB*C ⋅ C) - (2*BTx ⋅ C).
"""
function optimisation!(BTB::AbstractMatrix, BTx::AbstractArray, 
                        optimisation_method::Exact, _::Any)
    return inv(BTB)BTx
end

function optimisation!(BTB::AbstractMatrix, BTx::AbstractArray,
                            optimisation_method::Nesterov, qd::QuantizerData)
    α = 1/qd.n_dp
    C = if typeof(BTB) <: CuArray CuArray(qd.C) else qd.C end      
    dist=1
    i = 0
    μ = zero(C)
    C_old = zero(C)
    while dist > 1e-3
        C_old .= C
        β = 1-(3/(i+5))
        μ.*=β
        temp_C = C .- μ
        grad = gradient(x->quantization_objective(BTB, BTx, x), temp_C)[1]
        μ .+= α.*grad 
        C .-= μ
        C_old .-= C
        dist = norm(C_old)
        i+=1
    end
    return C
end
"""
    function update_codebook!(data::AbstractMatrix, qd::QuantizerData, update_method::UpdateMethod, 
    processing::Processing, optimisation_method::OptimizationMethod)
    
Main codebook update function; selects the appropriate methods to compute the ∑BTB and ∑BTx matrices
given the configurations and updates the codebook `qd.C` in place using the optimisation method selected.

"""
function update_codebook!(data::AbstractMatrix, qd::QuantizerData, update_method::BMatrix, 
                                processing::Processing, optimisation_method::OptimizationMethod)
    BTB, BTx = compute_BMatrices(data, qd, processing)
    qd.C[:] .= optimisation!(BTB, BTx, optimisation_method, qd)
end

######################################## IVF methods #############################################

function update_codebook!(data::AbstractMatrix, qd::QuantizerData, update_method::InvertedIndex,
                            processing::SingleThreaded, optimisation::Exact)
    for j in 1:qd.n_codebooks
        for k in 1:qd.n_centers
            C_i1 = ((j-1) * qd.n_dims_center * qd.n_centers) + ((k-1) * qd.n_dims_center) + 1
            C_i2 = ((j-1) * qd.n_dims_center * qd.n_centers) + ((k) * qd.n_dims_center)
            idxs = qd.I.IVF[j][k]
            if length(idxs) > 0
                qd.C[C_i1:C_i2] = dropdims(mean(@inbounds(@view(data[((j-1) * qd.n_dims_center + 1):j*qd.n_dims_center, idxs])),dims=2), dims=2)
            end
        end
    end
end

function update_codebook!(data::AbstractMatrix, qd::QuantizerData, update_method::InvertedIndex,
                            processing::MultiThreaded, optimisation::Exact)
    Threads.@threads for j in 1:qd.n_codebooks
        for k in 1:qd.n_centers
            C_i1 = ((j-1) * qd.n_dims_center * qd.n_centers) + ((k-1) * qd.n_dims_center) + 1
            C_i2 = ((j-1) * qd.n_dims_center * qd.n_centers) + ((k) * qd.n_dims_center)
            idxs = qd.I.IVF[j][k]
            filter!(x->x>1,idxs)        # Multi-threading instability seems to have a very small probability of 
            filter!(x->x<qd.n_dp,idxs)  # allocating random datapoint indexes
            if length(idxs) > 0
                qd.C[C_i1:C_i2] = dropdims(mean(@inbounds(@view(data[((j-1) * qd.n_dims_center + 1):j*qd.n_dims_center, idxs])),dims=2), dims=2)
            end
        end
    end
end


#################################################################################################
#                                                                                               #
#                                        Anisotropic loss                                       #
#                                                                                               #
#################################################################################################

compute_inner_matrix(dp::AbstractArray, η::AnisotropicWeights) = (η.h_par - η.h_orthog)*(dp*dp')+η.h_orthog*I(length(dp))

function compute_BTx(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights)
    sumBTx = zeros(qd.n_dims_center * qd.n_codebooks * qd.n_centers)
    for i in 1:qd.n_dp
        sumBTx+= (η.h_par*qd.I.B[i])*@view(data[:,i])
    end
    return sumBTx
end

function compute_BTB(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights, threading::SingleThreaded)
    Bsize = (qd.n_dims_center * qd.n_codebooks * qd.n_centers)
    sumBTB = zeros(Float32, Bsize, Bsize)
    for i in 1:qd.n_dp
        inner = compute_inner_matrix(@view(data[:,i]), η)
        left_mul = qd.I.B[i]*inner
        a,b,c = findnz(sparse(left_mul))
        a = unique(a)
        k = 0
        for i in a
            for j in a
                k+=1
                sumBTB[i,j] += c[k]
            end
        end
    end
    return sum(sumBTB,dims=3)
end

function compute_BTB(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights, threading::MultiThreaded)
    Bsize = (qd.n_dims_center * qd.n_codebooks * qd.n_centers)
    sumBTB = zeros(Float32, Bsize, Bsize, Threads.nthreads())
    Threads.@threads for i in 1:qd.n_dp
        inner = compute_inner_matrix(@view(data[:,i]), η)
        left_mul = qd.I.B[i]*inner
        a,b,c = findnz(sparse(left_mul))
        a = unique(a)
        k = 0
        for i in a
            for j in a
                k+=1
                sumBTB[i,j,Threads.threadid()] += c[k]
            end
        end
    end
    return dropdims(sum(sumBTB,dims=3), dims=3)
end


function compute_BMatrices(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights, processing::Processing)
    BTB = compute_BTB(data, qd, η, processing)
    BTx = compute_BTx(data, qd, η)
    return BTB, BTx
end

function compute_BMatrices(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights, processing::GPU)
    BTB = compute_BTB(data, qd, η, MultiThreaded())
    BTx = compute_BTx(data, qd, η)
    BTB = CuArray(BTB)
    BTx = CuArray(BTx)
    return BTB, BTx
end

function update_codebook!(data::AbstractMatrix, qd::QuantizerData, η::AnisotropicWeights,
                                processing::Processing, optimisation_method::OptimizationMethod)
    BTB, BTx = compute_BMatrices(data, qd, η, processing)
    try
        qd.C[:] .= optimisation!(BTB, BTx, optimisation_method, qd)
    catch
        @warn("""Singular Matrix encountered, some clusters do not contain assignments.
        This happens when the number of clusters is relatively high ccompared to the number of data points.
        \nOptimisation method changed to Approximate""")
        qd.C[:] .= optimisation!(BTB, BTx, Nesterov(), qd)
    end
end