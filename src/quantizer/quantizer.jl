include("initialization_step.jl")
include("partition_assignment_step.jl")
include("codebook_update_step.jl")

function quantizer(data::AbstractMatrix, η::L2_loss, codebook::Any, config::NamedTuple)
    if config[:verbose] @info("Starting L2 Quantization...") end
    thread, update_method, optim_method, processing = generate_config_vars(config)
    qd = GenerateQuantizerData(data, config.n_codebooks, config.n_centers)
    initialization!(data, qd, thread, codebook)
    dist = 1
    i = 0
    while (dist > config[:stopcond]) && (i < config[:max_iter])
        i += 1
        C_old = deepcopy(qd.C)
        iteration_loss = assignment_step!(data, qd, update_method, thread)
        update_codebook!(data, qd, update_method, processing, optim_method)
        dist = sqeuclidean(C_old, qd.C)
        if config[:verbose] && config[:multithreading] @info("Update distance: $dist")
        elseif config[:verbose] @info("Iteration loss: $iteration_loss\tUpdate distance:$dist") end
    end
    if config[:verbose]
        if i < config[:max_iter] @info("Product quantization converged after $i iterations") 
        else @info("Training stopped after reaching max ($i) iterations") end
    end
    if config[:inverted_index] rebuild_Bmatrix!(qd) end
    return qd
end

function quantizer(data::AbstractMatrix, η::AnisotropicWeights, codebook::Any, config::NamedTuple)
    if config[:verbose] @info("Starting Anisotropic Quantization...") end
    thread, update_method, optim_method, processing = generate_config_vars(config)
    qd = GenerateQuantizerData(data, config.n_codebooks, config.n_centers)
    initialization!(data, qd, thread, codebook)
    assignment_step!(data, qd, AHPQ.BMatrix(), thread)
    dist = 1
    i = 0
    while (dist > config[:stopcond]) && (i < config[:max_iter])
        i += 1
        C_old = deepcopy(qd.C)
        iteration_loss = assignment_step!(data, qd, η, config[:max_iter_assignments], thread)
        update_codebook!(data, qd, η, processing, optim_method)
        dist = sqeuclidean(C_old, qd.C)
        if config[:verbose] && config[:multithreading] @info("Update distance: $dist")
        elseif config[:verbose] @info("Iteration loss: $iteration_loss\tUpdate distance:$dist") end
    end
    if config[:verbose]
        if i < config[:max_iter] @info("Product quantization converged after $i iterations") 
        else @info("Training stopped after reaching max ($i) iterations") end
    end
    return qd    
end

function logrange(x1, n) return (Int(floor(10^y)) for y in range(log10(x1)-n+1, log10(x1), length=n)) end
function subsample(n_dp, data) return data[:,shuffle!(collect(1:size(data)[2]))[1:n_dp]] end

function incremental_quantization(data::AbstractMatrix, η::Weights, config::NamedTuple)
    thread, update_method, optim_method, processing = generate_config_vars(config)
    tp = if config.training_points > 0 config.training_points else size(data)[2] end
    samplesizes = collect(logrange(tp, config[:increment_steps]))
    if config[:verbose] @info("\n\nFitting quantizer on $(samplesizes[1]) data points...") end
    qd = quantizer(subsample(samplesizes[1], data), η, 0, config)
    codebook = deepcopy(qd.C)

    for samplesize in samplesizes[2:end]
        if config[:verbose] @info("\n\nFitting quantizer on $samplesize data points...") end
        qd = quantizer(subsample(samplesize, data), η, codebook, merge(config, (;use_precomputed_codebook=true)))
        codebook = deepcopy(qd.C)
    end
    return qd
end