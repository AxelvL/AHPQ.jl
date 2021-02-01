include("initialization_step.jl")
include("partition_assignment_step.jl")
include("codebook_update_step.jl")

function l2_quantization(data::AbstractMatrix, qd::QuantizerData, config::NamedTuple)
    if config[:verbose] @info("Starting L2 Quantization...") end
    thread, update_method, optim_method, processing = generate_config_vars(config)
    initialization!(data, qd, thread)
    if config[:verbose] @info("Finished initialization step...") end

    dist = 1
    i = 0
    while (dist > config[:stopcond]) && (i < config[:max_iter])
        i += 1
        C_old = deepcopy(qd.C)
        iteration_loss = assignment_step!(data, qd, update_method, thread)
        update_codebook!(data, qd, update_method, processing, optim_method)
        dist = sqeuclidean(C_old, qd.C)
        if config[:verbose] && config[:multithreading] @info("Update distance: $dist")
        else @info("Iteration loss: $iteration_loss\tUpdate distance:$dist") end
    end
    if config[:inverted_index] @info("implement rebuildBMATRIXFUNCTION") end
    if config[:verbose]
        if i < config[:max_iter] @info("Product quantization converged after $i iterations") 
        else @info("Training stopped after reaching max ($i) iterations") end
    end
    return qd
end