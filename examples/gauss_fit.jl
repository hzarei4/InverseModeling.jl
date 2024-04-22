using InverseModeling, View5D, Noise

function test_gaussfit()
    # true_val = (i0 = 10.0, σ=Fixed([5.0, 6.0]), μ=[1.2, 3.1], offset=1.0) # Fixed()
    true_val = (i0 = 10.0, σ=[5.0, 6.0], μ=[1.2, 3.1], offset=1.0) # Fixed()
    sz = (40,40)
    forward, fit_parameters = gauss_model(sz, true_val)
    meas = poisson(forward(true_val))
    # meas = forward(true_val)
    # start_val = (i0 = 10, σ = [2.2180254609037426, 3.4734027671622583], μ = [0.5654755165137773, 1.4606503074960915], offset = 1.0)
    res, res_img, resv = gauss_fit(meas, verbose=true);
    # start_val = (i0 = 10, σ=[2.0, 2.0], μ=[1.8, 2.5], offset=0.0 ) # Fixed()
    # res, res_img, resv = gauss_fit(meas, start_val, verbose=true);
    @ve meas, res_img, (meas .- res_img)
end

