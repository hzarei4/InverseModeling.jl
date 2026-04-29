# gradient test
@testset "Modeling Core" begin
    a = rand(ComplexF32,10,10)
    b = rand(ComplexF32,10,10)
    m = abs.(a) .> 0.3
    v = a[m]

    lo(v1) = sum(abs2.(into_mask!(v1, m, b) .- 0.5))
    lo2(v1) = sum(abs2.(v1 .- 0.5))
    g = gradient(lo, v)[1]
    g2 = gradient(lo2, v)[1]
    @test g == g2

    sz = (10, 10)
    dat = collect(InverseModeling.gaussian_sep(sz))
    # start_params = (i0 = 1.0, σ=1.0, μ=(2.0,1.1), offset=0.0)
    start_params = gauss_start(dat, 0.2; has_covariance=false)

    forward, fit_parameters, get_fit_results = gauss_model(sz, start_params)
    fit_params, fixed_params, get_fit_results, stripped_params = InverseModeling.prepare_fit(start_params) #

    myval = forward(fit_params)
    lo3(v1) = sum(abs2.(forward(v1) .- dat))
    grad_im = gradient(lo3, fit_params)[1]  

    # using FiniteDifferences:
    mygrad = grad(central_fdm(5, 1), lo3, fit_params)[1]
    @test isapprox(mygrad, grad_im, rtol=1e-4)

    myval = 2.0
    tst_val = (tst1=Normalize(Positive(myval), 10.0), tst2=Fixed(15))
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    @test fixed_vals == InverseModeling.ComponentVector((tst2=15.0,))
    @test isapprox(forward(tst_start_vals), myval)
    @test isapprox(tst_start_vals.tst1, sqrt(myval/10))

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, myval)
    # @test isapprox(backward(fwd), tst_val)

end
