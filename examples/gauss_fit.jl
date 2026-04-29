using InverseModeling, View5D, Noise
using BenchmarkTools
using Random

function test_gaussfit()
    # true_val = (i0 = 10.0, σ=Fixed([5.0, 6.0]), μ=[1.2, 3.1], offset=1.0) # Fixed()
    true_val = (i0 = 10.0, σ=[5.0, 6.0], μ=[1.2, 3.1], offset=1.0) # Fixed()
    # true_val = (i0 = 10.0, σ=(5.0, 6.0), μ=(1.2, 3.1), offset=1.0) # Fixed()
    sz = (64, 64)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    @time perfect = forward(true_val); # 76 alloc 20 kB
    
    # @btime perfect = forward($true_val); # 9 µs
    Random.seed!(1234)
    @time meas = Float32.(poisson(Float64.(perfect)));
    # meas = forward(true_val)
    # start_val = (i0 = 10, σ = [2.2180254609037426, 3.4734027671622583], μ = [0.5654755165137773, 1.4606503074960915], offset = 1.0)
    @time res, res_img, resv = gauss_fit(meas, verbose=false, x_abstol = 0.01f0); # 166k alloc 27 Mb
    # using ProfileCanvas
    # @profview_allocs res, res_img, resv = gauss_fit(meas, verbose=false);
    @btime res, res_img, resv = gauss_fit($meas, x_abstol = 0.01f0); # 35 ms, old way: 10s, custom_grad: 16 ms
    # current custom grad: 11.283 ms (84467 allocations: 18.43 MiB)
    # after improvemetn 1: 9.583 ms (79555 allocations: 18.52 MiB)
    # bc overload: 7.135 ms (51829 allocations: 15.46 MiB)
    # specialized broadcast: 6.441 ms (47241 allocations: 15.35 MiB), only for multiplicative broadcasts!
    # after large cleanup: 6.854 ms (46576 allocations: 14.05 MiB)
    # improved optimization using fg!: 6.377 ms (46617 allocations: 13.94 MiB)
    # better start-vals and x_abstol: 3.835 ms (26988 allocations: 8.17 MiB)

    @time res, res_img, resv = gauss_fit(meas, noise_model=loss_poisson_pos, x_abstol = 0.01f0); 
    @btime res, res_img, resv = gauss_fit($meas, noise_model=loss_poisson_pos, x_abstol = 0.01f0); 
    #  16.882 ms (59103 allocations: 46.82 MiB)
    # better start_vals:  1.498 ms (5848 allocations: 4.55 MiB)

    @time res, res_img, resv = gauss_fit(meas, noise_model=loss_anscombe_pos, x_abstol = 0.01f0); 
    @btime res, res_img, resv = gauss_fit($meas, noise_model=loss_anscombe_pos, x_abstol = 0.01f0); 
    #  better start-vals: 2.358 ms (10350 allocations: 7.40 MiB)

    start_val = (i0 = 8.0, σ=[4.0, 7.0], μ=[1.0, 2.1], offset=1.8) # Fixed()
    res, res_img, resv = gauss_fit(meas, start_val, x_abstol = 0.01f0); # 
    @time res, res_img, resv = gauss_fit(meas, start_val, x_abstol = 0.01f0); # 3.771 ms (27558 allocations: 8.33 MiB)
    @btime res, res_img, resv = gauss_fit($meas, $start_val, x_abstol = 0.01f0); # 3.771 ms (27558 allocations: 8.33 MiB)

    # start_val = (i0 = 10, σ=[2.0, 2.0], μ=[1.8, 2.5], offset=0.0 ) # Fixed()
    # res, res_img, resv = gauss_fit(meas, start_val, verbose=true);

    @ve meas, res_img, (meas .- res_img)
end

function find_type_instability()
    using Cthulhu
    true_val = (i0 = 10.0, σ=[5.0, 6.0], μ=[1.2, 3.1], offset=1.0) # Fixed()
    sz = (64, 64)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    @time perfect = forward(true_val); 
    @code_warntype forward(true_val); 
    @descend forward(true_val)
end    

function code_stability_test()
    true_val = (i0 = 10f0, σ=[5f0, 6f0], μ=[1.2f0, 3.1f0], offset=1f0) # Fixed()
    sz = (32, 32)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    forward(true_val)
    @code_warntype forward(true_val)
    # fwd = fit_fct
    fit_params, fixed_params, get_fit_results, stripped_params = InverseModeling.prepare_fit(true_val, Float32) #

    function forward2(fit_params)::Array{Float32,2}
        function g(id) # an accessor function for the parameters
            InverseModeling.get_fwd_val(stripped_params[id], id, fit_params, fixed_params)
        end
        res = fit_fct(g) # calls fwd giving it the parameter-access function g
        return res
    end

    forward2(true_val)
    @code_warntype forward2(true_val)
    @descend forward2(true_val)
end

function test_return_type()
    function foo(x)::Int64
        return x(:h)
    end

    dat = (h=1,)
    function bar(x)
        return dat[x]
    end

    foo(bar)
    @code_warntype foo(bar)

end