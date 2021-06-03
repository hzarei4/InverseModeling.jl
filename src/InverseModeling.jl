module InverseModeling
using ComponentArrays
using Optim, Zygote
export create_forward, Fixed, Positive, loss

struct Fixed{T}
    data::T
end

struct Positive{T}
    data::T
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


# access a NamedTuple with a symbol (eg  :a)
get_val(val::Fixed, id, fit, non_fit) = getindex(non_fit, id)
get_val(val::Positive, id, fit, non_fit) = abs2.(getindex(fit, id))
get_val(val, id, fit, non_fit) = getindex(fit, id)

 # construct a named tuple from a dict
construct_named_tuple(d) = NamedTuple{Tuple(keys(d))}(values(d))

# split a NamedTuple with Fit and NonFit types into two 
 # ComponentArrays
function prepare_fit(vals, dtype=Float64)
    fit_dict = Dict() #ComponentArray{dtype}()
    non_fit_dict = Dict() #ComponentArray{dtype}()
    for (key, val) in zip(keys(vals), vals)
        if val isa Fixed
            non_fit_dict[key] = val.data
        elseif val isa Positive
            fit_dict[key] = sqrt.(val.data)
        else
            fit_dict[key] = val
        end
    end
    fit_named_tuple = construct_named_tuple(fit_dict)
    non_fit_named_tuple = construct_named_tuple(non_fit_dict)
    
    fit_params = ComponentArray(fit_named_tuple)
    fixed_params = ComponentArray(non_fit_named_tuple)

    function get_fit_results(res)
        # g(id) = get_val(getindex(fit_params, id), id, fit_params, fixed_params) 
        vals = Optim.minimizer(res)
        fwd = NamedTuple{keys(vals)}(collect(get_val(vals, id, vals, fixed_params) for id in keys(vals)))
        return vals, fwd
    end

    return fit_params, fixed_params, get_fit_results
end

gaussian_norm(data, fwd) = sum(abs2.(data.-fwd))

function create_forward(fwd, params, dtype=Float64)
    fit_params, fixed_params, get_fit_results = prepare_fit(params, dtype)

    function forward(fit)
        g(id) = get_val(getindex(params, id), id, fit, fixed_params) 
        return fwd(g)
    end
    
    return fit_params, fixed_params, forward, get_fit_results
end

function loss(data, forward, my_norm=gaussian_norm)
    return (params) -> my_norm(data, forward(params))
end

function optimize(loss_fkt, start_vals; iterations=100, optimizer=LBFGS())
    optim_options = Optim.Options(iterations=iterations)
    # g!(G,vec) = G.=gradient(loss_fkt,vec)[1]
    # @time optim_res = Optim.optimize(loss_fkt, g!, start_vals, optimizer, optim_options)
    @time optim_res = Optim.optimize(loss_fkt, start_vals, optimizer, optim_options) # ;  autodiff = :forward
    optim_res
end

function main()
    # using InverseModeling, Plots, Optim
    start_val = (σ=Positive([8.0, 25.0]),
            μ=Fixed([1.0, 3.0]) )

    times = -10:10
    fit_fkt(params) = (times.*params(:σ)[1]).^2 + (times.*params(:σ)[2]).^3 .+ times .* params(:μ)[1]

    start_vals, fixed_vals, forward, get_fit_results = create_forward(fit_fkt, start_val)
    meas = forward(start_vals)
    start_vals.σ += [0.5,0.2]

    @show optim_res = Optim.optimize(loss(meas, forward), start_vals)
    bare, fit_res = get_fit_results(optim_res)
    fit = forward(bare)

    # plot(times, forward(start_vals))
    # plot!(times, meas)
    # plot!(times, fit)
end

end # module
