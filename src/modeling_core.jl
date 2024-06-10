export create_forward, sim_forward, loss, optimize_model
export get_loss
export Fixed, Positive, Normalize, ClampSum
export into_mask!
export sum!_

# this datatyp specifies that this parameter is not part of the fit
struct Fixed{T}
    data::T
end

# this datatyp specifies that this parameter is kept positive during the fit
# this is achieved by introducing an auxiliary function whos abs2.() yields the parameter
struct Positive{T}
    data::T
end

"""  Normalize{T}
    this datatyp specifies that this parameter is normalized before the fit 
    this is achieved by deviding by the factor in the inverse path and multiplying by it in the forward model.
    Note that by chosing a fit parameter `p` for example using `Normalize(p, maximum(p))` or `Normalize(p, mean(p))`, the fit-variable will be unitless,
    but the result will automatically be cast back to the original scale. This helps the fit to converge.
"""
struct Normalize{T,F}
    data::T
    factor::F
end

""" ClampSum{T}
    this contraint ensures that the value will never change its sum (optionally over predifined dimensions) during optimization. This is useful to avoid ambiguities during optimization.
    In practice the last value is simply computed by the given initial sum minus all other values.
"""
struct ClampSum{T, S, N}
    data::T
    mysum::S
    mysize::NTuple{N, Int}
end
""" ClampSum(dat::T, dims=ntuple((n)->n, ndims(dat))) where {T}

convenience constructor. 
# Arguments
+ `data`: data for which the sum should be clamped
+ `dims`: defines the dimensions over which to sums should be clamped. The default is all dimensions (clamps the total sum)
"""
function ClampSum(dat::T) where {T}
    val = get_val(dat)
    mysum = sum(val)
    ClampSum{T,typeof(mysum),length(size(val))}(dat, mysum, size(val))
end

"""
    @my_get t s

Access a NamedTuple/Struct `t` with the field `s`.
```
julia> t = (a=10, b=12.0)
(a = 10, b = 12.0)

julia> @my_get t a
10
```
"""
macro my_get(t, x)
    :($t.$x)
end

# if Fixed appears anywhere in a chain of modifyers, all of them are ignored
is_fixed(val) = false
is_fixed(val::Fixed) = true
is_fixed(val::Positive) = is_fixed(val.data)
is_fixed(val::Normalize) = is_fixed(val.data)
is_fixed(val::ClampSum) = is_fixed(val.data)

# access a NamedTuple with a symbol (eg  :a)
# this returns just the bare value
get_val(val) = val
get_val(val::Fixed) = get_val(val.data) # to strip off the properties
get_val(val::Positive) = get_val(val.data) # to strip off the properties
get_val(val::Normalize) = get_val(val.data) # to strip off the properties
get_val(val::ClampSum) = get_val(val.data) # to strip off the properties

# evaluated the pre-forward model modifyers
get_fwd_val(val) = val
get_fwd_val(val::Fixed) = get_fwd_val(val.data) # to strip off the properties
get_fwd_val(val::Positive) = abs2.(get_fwd_val(val.data)) 
get_fwd_val(val::Normalize) = get_fwd_val(val.data) .* val.factor
function get_fwd_val(val::ClampSum)
    myvals = get_fwd_val(val.data)
    # @show val.mysum
    # @show sum(myvals)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end

get_fwd_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id) # do not apply any modifiers here.
get_fwd_val(val::Positive, id, fit, non_fit) = abs2.(get_fwd_val(val.data, id, fit, non_fit))
get_fwd_val(val::Normalize, id, fit, non_fit) = get_fwd_val(val.data, id, fit, non_fit) .* val.factor
function get_fwd_val(val::ClampSum, id, fit, non_fit)
    myvals = get_fwd_val(val.data, id, fit, non_fit)
    # @show val.mysum
    # @show sum(myvals)
    reshape(vcat(myvals, val.mysum - sum(myvals)), val.mysize)
end
get_fwd_val(val, id, fit_params, non_fit) = get_fwd_val(@view fit_params[id])
# begin 
#     tmp = @view fit_params[id]#  getindex(fit, id)
#     get_fwd_val(tmp)
# end

# inverse opterations for the pre-forward model parts
get_inv_val(val, fct=identity) = fct.(val)
get_inv_val(val::Fixed, fct=identity) = get_inv_val(val.data, fct)
get_inv_val(val::Positive, fct=identity) = get_inv_val(val.data, (x)->(sqrt.(fct.(x))))
get_inv_val(val::Normalize, fct=identity) = get_inv_val(val.data, (x)->fct.(x)./ val.factor) 
get_inv_val(val::ClampSum, fct=identity) = get_inv_val(val.data, (x)->clamp_sum(fct.(x)))
function clamp_sum(val)
    myval = get_inv_val(val)
    return myval[1:end-1] # optimize the 1D- view with the last value missing
end

# function truncated_view(dat, dims)
#     return @view dat[(1:ifelse(d in dims, size(dat,d)-1, size(dat, d)) for d in ndims(dat))...]
# end

 # construct a named tuple from a dict
construct_named_tuple(d) = NamedTuple{Tuple(keys(d))}(values(d))

# split a NamedTuple with Fit and NonFit types into two 
# ComponentArrays
"""
    prepare_fit(vals, dtype=Float64)

this function is called before a fit is started.
#arguments
+ `vals`:   values to compare the data with
# `dtype`:  currently unused

#returns
a tuple of 
`fit_params`    : the (variable) fit parameters
`fixed_params`  : the fixed parameters with which the model is called but they are not optimized for
`get_fit_results`: a function that retrieves a tuple `(bare, params)` of the preforwardmodel fit result and a named tuple of version when called with the result of `optim()`
                `bare` is the raw result withoút the pre_forward_model being applied. This can be plugged into `foward`
                `param` is a named tuple of the result according to the model that the user specified.  
`stripped_params``: a version of params, with the fixed parameters stripped from all modifications
"""
function prepare_fit(vals, dtype=Float64)  # 
    fit_dict = Dict() #ComponentArray{dtype}()
    non_fit_dict = Dict() #ComponentArray{dtype}()
    stripped_params = Dict() # removed all modifications from Fixed values
    for (key, val) in zip(keys(vals), vals)        
        if is_fixed(val) # isa Fixed
            non_fit_dict[key] = get_val(val) # all other modifiers are ignored
            stripped_params[key] = Fixed(get_val(val)) 
        else
            fit_dict[key] = get_inv_val(val)
            stripped_params[key] = val
        end
    end
    fit_named_tuple = construct_named_tuple(fit_dict)
    non_fit_named_tuple = construct_named_tuple(non_fit_dict)
    stripped_params = construct_named_tuple(stripped_params)
    
    fit_params = ComponentArray{dtype}(fit_named_tuple) # the optim routine cannot deal with tuples
    fixed_params = ComponentArray{dtype}(non_fit_named_tuple)

    function get_fit_results(res)
        # g(id) = get_val(getindex(fit_params, id), id, fit_params, fixed_params) 
        bare = res.minimizer # Optim.minimizer(res)
        all_keys = keys(bare)
        # The line below may apply pre-forward-models to the fit results. This is not necessary for the fixed params
        fwd = NamedTuple{all_keys}(collect(get_fwd_val(vals[id], id, bare, fixed_params) for id in keys(bare)))
        fwd = merge(fwd, fixed_params)
        return bare, fwd
    end

    return fit_params, fixed_params, get_fit_results, stripped_params
end

"""
    create_forward(fwd, params)

creates a forward model given a model function `fwd` and a set of parameters `param`.
The properties such as `Positive` or `Normalize` of the modified `params` are baked into the model
#returns
a tuple of
`fit_params`    : a collection of the parameters to fit
`fixed_params`  : a collection of the fixed parameters exluded from the fitting, but provided to the model
`forward`       : the forward model
`backward`      : the adjoint model
`get_fit_results`: a function that retrieves the fit result for the result of optim
"""
function create_forward(fwd, params, dtype=Float32) # 
    fit_params, fixed_params, get_fit_results, stripped_params = prepare_fit(params, dtype) #

    # can be called with a NamedTuple or a ComponentArray. This will call the fwd function, 
    # which itself needs to access its one argument by function calls with the ids
    # fwd(g) which accesses the parameters via g(:myparamname)
    function forward(fit_params)
        # g(id) = get_val(getindex(stripped_params, id), id, fit, fixed_params) 
        # g is a function which receives an id and returns the corresponding paramter.
        # id corresponds to the variable names in a named tuple or ComponentArray
        function g(id)
            #v = get_fwd_val(stripped_params[id], id, fit, fixed_params) 
            #@show stripped_params[id]
            #@show v
            # @show id
            #println("params[$(id)] is $(params[id])")
            #println("$(id) is $(fit[id]) is $v")
            # v
            # DO NOT PUT @time HERE! This will break Zygote.
            get_fwd_val(stripped_params[id], id, fit_params, fixed_params) 
        end
        return fwd(g) # calls fwd giving it the parameter-access function g
    end

    function backward(vals)
        all_keys = keys(vals)
        NamedTuple{all_keys}(collect(get_fwd_val(vals, id, vals, fixed_params) for id in keys(vals)))
    end
    
    return fit_params, fixed_params, forward, backward, get_fit_results
end

"""
    sim_forward(fwd, params)

creates a model with a set of parameters and runs the forward method `fwd` to obtain the result.
This is useful for a simulation. No noise is applied, but can be applied afterwards.
"""
function sim_forward(fwd, params)
    vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, params)
    return forward(vals);
end

"""
    loss(data, forward, my_norm=norm_gaussian)

returns a loss function given a forward model `forward` with some measured data `data`. The noise_model is specified by `my_norm`.
The returned function needs to be called with parameters to be given to the forward model as arguments. 
"""
function loss(data, forward, my_norm = loss_gaussian, bg=eltype(data)(0))
    return (params) -> my_norm(data, forward(params), bg)
end

"""
    optimize_model(loss_fkt, start_vals; iterations=100, optimizer=LBFGS(), kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.
Other options such as `store_trace=true` can be provided and will be passed to `Optim.Options`.

#arguments
+ `loss_fkt`        : the loss function to optimize
+ `start_vals`      : the set of parameters over which the optimization is performed
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result as provided by Optim.optimize()
"""
function optimize_model(loss_fkt::Function, start_vals; iterations=100, optimizer=LBFGS(), kwargs...)
    optim_options = Optim.Options(;iterations=iterations, kwargs...)
    g!(G,vec) = G.=gradient(loss_fkt,vec)[1]
    @time optim_res = Optim.optimize(loss_fkt, g!, start_vals, optimizer, optim_options)
    # optim_res = Optim.optimize(loss_fkt, start_vals, optimizer, optim_options) # ;  autodiff = :forward
    optim_res
end

"""
    optimize_model(start_val::Tuple, fwd_model::Function, loss_type=loss_gaussian; iterations=100, optimizer=LBFGS(), store_trace=true, kwargs...)

performs the optimization of the model parameters by calling Optim.optimize() and returns the result.

#arguments
+ `start_val`: the set of parameters over which the optimization is performed
+ `fwd_model`: the model which is optimized.
+ `loss_type`: the type of the loss function to use.
+ `iterations`      : number of iterations to perform (default: 100). This is provided via the `Optim.Options` stucture.
+ `optimizer=LBFGS()`: the optimizer to use

#returns
the result is a Tuple of `res` and the trace of the loss function value. `res` is a `ComponentArray` with all the results after applying the pre-forward part of the algorithm.
This includes the values marked as `Fixed()`.
if the argument ``store_trace=false` is provided no trace will be returned.

#See also:
The other (low-level) version of `optimize_model` with the loss function as the first argument.
"""
function optimize_model(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian; iterations=100, optimizer=LBFGS(), store_trace=true, kwargs...)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);
    optim_res = InverseModeling.optimize_model(loss(meas, forward, loss_type), start_vals; iterations=iterations, optimizer=optimizer, store_trace=store_trace, kwargs...);
    bare, res = get_fit_results(optim_res)
    if store_trace
        return res, [t.value for t in optim_res.trace][2:end]
    else
        return res
    end
end

function get_loss(start_val::NamedTuple, fwd_model::Function, meas, loss_type=loss_gaussian)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_model, start_val);
    loss(meas, forward, loss_type)(start_vals)
end

"""
    into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}

fills a vector into a mask `mymask`. This is very useful for optimizing only some pixels in an image. E.g. the inner part of a Fourier-space aperture.
#Arguments
+ `vec` : the vector to fill into the mask
+ `mymask`  : a binary mask to use
+ `tmp_data` : the mask is filled into only this part of the dataset. For efficiency reasons you should provide this argument, even if this is not strictly necessary as zeros are filled in.
            Note that you can also supply this argument to have a part of the resulting image being fixed.
#Returns
tmp_mask
"""
function into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}
    tmp_data[mymask] = vec
    return tmp_data
end

using ChainRulesCore
# define custom adjoint for, since mutating arrays is not supported
function ChainRulesCore.rrule(::typeof(into_mask!), vec::AbstractArray{T, 1}, amask::AbstractArray{Bool, M},
    tmp_data=zeros(eltype(vec), size(amask))) where {T, M}
    Y = into_mask!(vec, amask, tmp_data)
    function into_mask_pullback(barx)
        if eltype(vec) <: Complex
            return NoTangent(), barx[amask], NoTangent(), NoTangent()
        else
            return NoTangent(), real(barx[amask]), NoTangent(), NoTangent()
        end
    end
    return Y, into_mask_pullback
end

"""
    sum!_(mymem, accum!, N)

A quick and dirty sum! function which sets mymem first to zero and then calls the `accum!` function.
A total of N accumulations are executed in place on memory `mymem`.

This function was mainly written to be differentiable via an rrule.

# Arguments
+ `accum!` expects mymem as the first argument and an index `n` as the second.

"""
function sum!_(mymem, accum!, N)
    mymem .= zero(eltype(mymem))
    for n=1:N
        accum!(mymem, n)
    end
    return mymem
end

function ChainRulesCore.rrule(::typeof(sum!_), mymem, accum!, N)
    Y = sum!_(mymem, accum!, N)
    function sum!__pullback(y)
        @show "pullback"
        @show size(mymem)
        @show size(y)
        return NoTangent(), y .* mymem, NoTangent(), NoTangent()
    end
    return Y, sum!__pullback
end
