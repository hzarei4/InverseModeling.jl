module InverseModeling
using ComponentArrays
using IndexFunArrays
using Optim, Zygote

include("noise_models.jl")
include("modeling_core.jl")
include("model_gauss.jl")

end # module
