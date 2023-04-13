# using InverseModeling, View5D, Noise, NDTools, FourierTools
using View5D, NDTools, FourierTools, TestImages, IndexFunArrays, ComponentArrays
using InverseModeling
using Plots
using Zygote, Noise
using PointSpreadFunctions

function test_ism_recon()

    obj = select_region(Float32.(testimage("resolution_test_512")), new_size=(128,128), center=(249,374)) .* 1000;
    sz = size(obj)

    pp_ex = PSFParams(0.488, 1.4, 1.52, pol=pol_scalar);
    p_ex = psf(sz, pp_ex; sampling=(0.050,0.050,0.200));

    # make a different (aberrated) psf
    aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[1.5, 0.6])
    # aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[0.5, 0.6])
    pp_em = PSFParams(0.520, 1.4, 1.52, pol=pol_scalar, method=MethodPropagateIterative, aberrations=aberrations);
    a_em = apsf(sz, pp_em; sampling=(0.050,0.050,0.200));
    p_em2 = abs2.(a_em)
    # @vp ft(a_em)
    p_em = psf(sz, pp_em; sampling=(0.050,0.050,0.200));

    # @vt p_ex p_em p_em2

    psf_ex = ifftshift(p_ex)
    asf_em = ifftshift(a_em)[:,:,1,1]
    pupil_em = fft(asf_em)
    pupil_noabber = abs.(pupil_em) .+0im
    psf_em = ifftshift(p_em)

    dx = 1.1; dy = 1.1;
    all_shifts = Tuple(ifftshift(exp_ikx(sz,shift_by=(x*dx,y*dy))) for x in -2:2, y in -2:2);
    all_shifts = cat(all_shifts..., dims=4);

    # pupil reconstruction model
    pupil_mask = (abs.(pupil_em) .> 0.1)
    pupil_mask[1] = false # fix the middle pixel to remove ambiguities
    modified_pupil = copy(pupil_noabber)
    function fwd_amp(params)
        # its important to have the return value for the gradient to compute
        mod_pupil = into_mask!(params(:pupil_em), pupil_mask, modified_pupil)
        psf_em = abs2.(ifft(mod_pupil))
        all_psfs = psf_ex .* real.(ifft(fft(psf_em) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    pupil_strength = abs.(pupil_noabber[pupil_mask])
    function fwd_phase(params)
        pupil_vals = pupil_strength .* cis.(params(:phase_em))
        mod_pupil = into_mask!(pupil_vals, pupil_mask, modified_pupil)
        psf_em = abs2.(ifft(mod_pupil))
        all_psfs = psf_ex .* real.(ifft(fft(psf_em) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # intensity psf reconstruction model
    function fwd_int(params)
        all_psfs = psf_ex .* real.(ifft(fft(params(:psf_em)) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # make a simulation
    meas = sim_forward(fwd_amp, (obj=Positive(obj), pupil_em=Fixed(pupil_em[pupil_mask])));
    # meas = Noise.poisson(meas);
    mymean = sum(meas) ./ prod(size(meas))
    mymax = maximum(meas) 
    numpix = prod(size(meas))

    # start with some object-only iterations
    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean),  4.0*mymean), psf_em=Fixed(psf_ex))
    start_val = (obj=Normalize(Positive(res1[:obj]), 16.0*mymean), phase_em=Fixed(angle.(pupil_noabber[pupil_mask])))
    # res1, myloss1 = optimize_model(start_val, fwd_int, meas; iterations=50)
    res1, myloss1 = optimize_model(start_val, fwd_phase, meas; iterations=50)
    print("Loss is $(myloss1[end])")
    plot(myloss1, label="L-BFGS") # , yaxis=:log

    @vt obj res1[:obj]

    recon_int = false
    phase_only = true
    myfwd = nothing
    if recon_int
        start_val = (obj=Normalize(Positive(ones(sz) .* mymean),  4.0*mymean), psf_em=Positive(psf_ex))
        myfwd = fwd_int
        # start_val = (obj=Fixed(res1[:obj]), psf_em=Positive(psf_ex))
        # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Positive(psf_ex))
        # psfmean = sum(psf_ex) ./ psf_ex
        # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Normalize(Positive(psf_ex), psfmean))
        # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), numpix), psf_em=ClampSum(Positive(psf_ex)))
    else
        if phase_only
            start_val = (obj=Normalize(Positive(res1[:obj]), 16.0*mymean), phase_em=angle.(pupil_noabber[pupil_mask]))
            # start_val = (obj=Fixed(res1[:obj]), phase_em= angle.(pupil_noabber[pupil_mask]))
            # start_val = (obj=Positive(res1[:obj]), phase_em= angle.(pupil_noabber[pupil_mask]))
            # start_val = (obj=Fixed(obj), phase_em= angle.(0.1 .*rand(size(pupil_noabber[pupil_mask]))))
            # start_val = (obj=Fixed(obj), phase_em= angle.(pupil_em[pupil_mask]))
            myfwd = fwd_phase
        else
            # start_val = (obj=Normalize(Positive(ones(sz) .* mymean),  4.0*mymean), pupil_em=pupil_noabber[pupil_mask])
            # start_val = (obj=Normalize(Positive(res1[:obj]),  4.0*mymean), pupil_em=pupil_noabber[pupil_mask])
            start_val = (obj=Positive(res1[:obj]), pupil_em=pupil_noabber[pupil_mask])
            myfwd = fwd_amp
        end
    end
    get_loss(start_val, myfwd, meas)

    res, myloss = optimize_model(start_val, myfwd, meas; iterations=450)
    println("Loss is $(myloss[end])")
    plot!(myloss, label="L-BFGS") # , yaxis=:log
    if phase_only
        into_mask!(pupil_strength .* cis.(res[:phase_em]), pupil_mask, modified_pupil)
    else
        into_mask!(res[:pupil_em], pupil_mask, modified_pupil)
    end
    @vtp fftshift(pupil_em) fftshift(modified_pupil)

    @vtp fftshift(psf_em) fftshift(res[:psf_em])


    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Fixed(psf_em))
    # start_val = (obj=Positive(ones(sz) .* mymean), psf_em=Fixed(psf_em))
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(myfwd, start_val);
    size(forward(start_vals)) 
    # @vt forward(start_vals) meas

    @time first_pred = forward(start_vals)

    # # do some Zygote tests
    # a = rand(32,32)
    # b = rand(32,32,1,25)
    # m = rand(32,32,1,25)
    # # all_psfs = psf_ex .* real.(ifft(fft(params(:psf_em)) .* all_shifts, (1,2,3)))
    # myloss(a) = sum(abs2.(a .* real.(ifft(fft(a) .* b, (1,2,3))) .- m))
    # @time gradient(myloss, a)[1];
    loss(meas, forward)(start_vals)
    @time gradient(loss(meas, forward), start_vals)[1]


    @time optim_res = InverseModeling.optimize_model(loss(meas, forward, loss_gaussian), start_vals, iterations=55, store_trace=true);
    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    print("Loss is $(optim_res.minimum)")
    plot!([t.iteration for t in optim_res.trace][2:end], [t.value for t in optim_res.trace][2:end], label="L-BFGS") # , yaxis=:log
    bare, res = get_fit_results(optim_res)

    # @ve obj, res[:obj]
    # into_mask!(res[:pupil_em], pupil_mask, modified_pupil)
    into_mask!(pupil_strength .* cis.(res[:phase_em]), pupil_mask, modified_pupil)
    @vtp fftshift(pupil_em) fftshift(modified_pupil)

    g = gradient(loss(meas, forward), start_vals)[1]
    gr = into_mask!(g, pupil_mask)
    @vtp fftshift(gr)

    @vt fftshift(psf_em) fftshift(res[:psf_em])
    plot(psf_ex[1:10,1], label="psf_ex")
    plot!(psf_em[1:10,1], label="GT psf_em")
    plot!(res[:psf_em][1:10,1], label="recovered psf_em")


    # gradient test
    a = rand(ComplexF32,10,10)
    b = rand(ComplexF32,10,10)
    m = abs.(a) .> 0.3
    v = a[m]

    lo(v1) = sum(abs2.(into_mask!(v1, m, b) .- 0.5))
    lo2(v1) = sum(abs2.(v1 .- 0.5))
    g = gradient(lo, v)[1]
    g2 = gradient(lo2, v)[1]
    g == g2

    #
    tst_val = (tst1=Normalize(Positive(2.0), 10.0),)
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    forward(tst_start_vals)

end

