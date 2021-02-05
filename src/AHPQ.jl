module AHPQ
    using LinearAlgebra
    using SparseArrays               # For memory-efficient identity matrices
    using Random: shuffle!           # For random initialisation
    using CUDA: CuArray              # For GPU Processing
    using Zygote: gradient           # For fast gradient computation
    using Statistics: mean, norm     # For fast stats computation

    include("utils/configs.jl")
    include("utils/distances.jl")
    include("quantizer/quantizer.jl")
    include("quantizer/preclustering.jl")
    include("builder.jl")
    include("searcher.jl")

    export builder
    export searcher

end
