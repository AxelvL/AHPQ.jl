include("initialization_step.jl")
include("partition_assignment_step.jl")
#include("optimization_step.jl")

function l2_quantization(data::AbstractMatrix, qd::QuantizerData, config::NamedTuple)
    if config[:verbose]
        @info("Starting reconstruction loss quantizer...")
    end
    config.multithreading ? threading = MultiThreaded() : threading = SingleThreaded()
    initialization!(data, qd, threading)
    if config[:verbose]
        @info("Finished initialization step...")
    end
    dist = 1
    i = 0
    while (dist > config[:stopcond]) && (i < config[:max_iter])
        i += 1
        C_old = deepcopy(qd.C)
    end
    return qd
end