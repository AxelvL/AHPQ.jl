struct SingleThreaded end
struct MultiThreading end 
struct CPU end
struct GPU end
struct Exact end
struct Nesterov end

const DEFAULT_CONFIG = (n_codebooks=0,
                        n_centers=8,
                        a=0,
                        b=0,
                        initialize_with_euclidean_loss=true, 
                        inverted_index=true, 
                        max_iter=1000,
                        stopcond=9e-2,
                        verbose=false,
                        max_iter_assignments=10,
                        optimization="exact",
                        multithreading=false,
                        GPU=false,
                        increment_steps=4,
                        reorder                = 0,
                        training_points        = 0)

function generate_data_dependent_defaults(n_dp::Int, n_dims::Int)
    a = floor(Int,sqrt(n_dp))
    config_updated =   (n_codebooks = n_dims ÷ 2,
                        a = a,
                        b = (a ÷ 5) + 1,
                        training_points        = Int(ceil(n_dp/5)))
                        
    config = merge(DEFAULT_CONFIG, config_updated)

    return config
end

function check_kwargs(config, n_dims) 
    for key in keys(config)
        if key ∉ keys(DEFAULT_CONFIG)
            error("$key is not a valid configuration keyword")
        end
    end
    if n_dims % config.n_codebooks != 0
        error("Number of dimensions of data should be divisable by number of codebooks.
               Make sure to set the right n_codebook in the builder.")
    end
end

