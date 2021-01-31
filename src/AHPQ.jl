module AHPQ
    using LinearAlgebra
    using SparseArrays # For memory-efficient identity matrices
    using Random

    include("utils/configs.jl")
    include("quantizer/quantizer.jl")
    include("builder.jl")
    include("searcher.jl")

    export builder

end
