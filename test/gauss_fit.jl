using Random, Noise

function check_res(res, true_val, atol=1f-1, otol=0.8)
    @test isapprox(res[:i0], true_val[:i0], atol=atol)
    @test isapprox(res[:μ], true_val[:μ], atol=atol)
    @test isapprox(res[:σ], true_val[:σ], atol=atol)

    @test isapprox(res[:offset], true_val[:offset], atol=otol)
end

function test_gaus_fit(meas, true_val; atol=1f-1, otol=0.8, x_abstol = 0.01f0, start_val = true_val)
    res, res_img, resv = gauss_fit(meas, start_val, verbose=false, x_abstol = x_abstol); # 166k alloc 27 Mb
    check_res(res, true_val, atol, otol)

    res, res_img, resv = gauss_fit(meas, start_val, noise_model=loss_poisson_pos, x_abstol = x_abstol); 
    check_res(res, true_val, atol, otol)

    res, res_img, resv = gauss_fit(meas, start_val, noise_model=loss_anscombe_pos, x_abstol = x_abstol); 
    check_res(res, true_val, atol, otol)

    res, res_img, resv = gauss_fit(meas, start_val, x_abstol = x_abstol); # 
    check_res(res, true_val, atol, otol)
end

@testset "Gauss Fit" begin
    # true_val = (i0 = 10.0, σ=Fixed([5.0, 6.0]), μ=[1.2, 3.1], offset=1.0) # Fixed()
    true_val = (i0 = 10.0, σ=[5.0, 6.0], μ=[1.2, 3.1], offset=1.0) # Fixed()
    # true_val = (i0 = 10.0, σ=(5.0, 6.0), μ=(1.2, 3.1), offset=1.0) # Fixed()
    sz = (64, 64)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    perfect = forward(true_val); # 76 alloc 20 kB
    
    Random.seed!(1234)
    meas = Float32.(poisson(Float64.(perfect)));

    test_gaus_fit(perfect, true_val, x_abstol = 0.0001f0, otol=0.01); # 166k alloc 27 Mb

    test_gaus_fit(meas, true_val, x_abstol = 0.01f0, otol=0.6); # 166k alloc 27 Mb
end
