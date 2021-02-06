using SafeTestsets

@safetestset  "AHPQ.jl" begin
    include("builder.jl")
    include("searcher.jl")
    include("testmetrics.jl")
end
