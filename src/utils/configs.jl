struct SingleThreaded end
struct MultiThreaded end
struct GPU end
Processing = Union{SingleThreaded, MultiThreaded, GPU}
struct InvertedIndex end
struct BMatrix end 
UpdateMethod = Union{InvertedIndex, BMatrix}
struct Exact end
struct Nesterov end
OptimizationMethod = Union{Exact, Nesterov}

const DEFAULT_CONFIG = (n_codebooks=0,
                        n_centers=8,
                        a=0,
                        b=0,
                        initialise_with_euclidean_loss=true, 
                        inverted_index=false, 
                        max_iter=1000,
                        stopcond=1e-2,
                        verbose=true,
                        max_iter_assignments=10,
                        optimisation="exact",
                        multithreading=false,
                        GPU=false,
                        increment_steps=0,
                        incremental_fitting_iteration=false,
                        reorder                = 0,
                        training_points        = 0)

function generate_data_dependent_defaults(n_dp::Int, n_dims::Int)
    a = floor(Int,sqrt(n_dp))
    training_points = Int(ceil(n_dp/5))
    config_updated  =   (n_codebooks = n_dims ÷ 2,
                        a = a,
                        b = (a ÷ 5) + 1,
                        training_points        = training_points,
                        increment_steps=Int(floor(log10(training_points))-2))
    config = merge(DEFAULT_CONFIG, config_updated)
    return config
end

function check_kwargs(kwargs, n_dp, n_dims) 
    for key in keys(kwargs)
        if key ∉ keys(DEFAULT_CONFIG)
            error("$key is not a valid configuration keyword")
        end
    end
    config = generate_data_dependent_defaults(n_dp, n_dims)
    config = merge(config, kwargs)
    if n_dims % config.n_codebooks != 0
        error("Number of dimensions of data should be divisable by number of codebooks.
               Make sure to set the right n_codebook in the builder.")
    end
    tp = if config.training_points > 0 config.training_points else n_dp end
    if config.increment_steps > 0
        smallest_sample = collect(logrange(tp, config.increment_steps))[1]
        if !(smallest_sample > config.a) 
            @warn("Lowest subsample size for incremental fitting is lower than number of centers. Increment_steps have been decreased.")
            config = merge(config, (; increment_steps = Int(ceil(log10(tp))-ceil(log10(config.a)))))
        end
    end
    if tp < config.n_centers
        @error("$tp Training points is lower than $(config.n_centers)")
    end
    if config.GPU 
        config = merge(config, (;optimisation="Nesterov")) 
        @warn("GPU processing was selected, codebook optimisation has automatically changed to approximate method.")
    end

    return config
end

function generate_config_vars(config)
    thread = if config.multithreading MultiThreaded() else SingleThreaded() end
    update_method = if config.inverted_index InvertedIndex() else BMatrix() end
    optim_method = if lowercase(config.optimisation)[1]=='e' Exact() else Nesterov() end
    processing = if config.GPU GPU() else thread end
    return thread, update_method, optim_method, processing
end

struct AnisotropicWeights{F<:AbstractFloat}
    η::F
    h_orthog::F
    h_par::F
end

struct L2_loss end
Weights = Union{AnisotropicWeights, L2_loss}

compute_η(T::Real, n_dim::Int) = (n_dim-1)*T^2/(1-T^2)

function ComputeWeightsFromT(n_dim::Int, T::AbstractFloat)
    η = compute_η(T, n_dim)
    h_orthog = 1/(1+η)
    h_par    = 1 - h_orthog
    return AnisotropicWeights(η, h_orthog, h_par)
end