# using InverseModeling, View5D, Noise, NDTools, FourierTools
using View5D, NDTools, FourierTools, TestImages
using IndexFunArrays, ComponentArrays
using InverseModeling
using Plots
using Noise
using PointSpreadFunctions
# using Zygote

function test_ism_recon()

    aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[1.5, 0.6])
    λ_ex = 0.488
    λ_em = 0.520
    sampling = (0.050,0.050,0.200)

    obj = select_region(Float32.(testimage("resolution_test_512")), new_size=(128,128), center=(249,374)) .* 50000;
    sz = size(obj)

    pp_ex = PSFParams(λ_ex, 1.4, 1.52, pol=pol_scalar, method=MethodPropagateIterative, aberrations=aberrations);
    p_ex = psf(sz, pp_ex; sampling=sampling);

    # make a different (aberrated) psf
    # aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[0.5, 0.6])
    pp_em = PSFParams(λ_em, 1.4, 1.52, pol=pol_scalar, method=MethodPropagateIterative, aberrations=aberrations);
    a_em = apsf(sz, pp_em; sampling=sampling);
    p_em = psf(sz, pp_em; sampling=sampling);

    # @vt p_ex p_em p_em2

    psf_ex = ifftshift(p_ex)
    asf_em = ifftshift(a_em)[:,:,1,1]
    pupil_em = fft(asf_em)
    pupil_noabber = abs.(pupil_em) .+0im
    psf_em = ifftshift(p_em)
    psf_noabber = abs2.(ifft(pupil_noabber))

    # make shif factors (in Fourier space)
    dx = 1.1; dy = 1.1;
    all_shifts = Tuple(ifftshift(exp_ikx(sz,shift_by=(x*dx,y*dy))) for x in -2:2, y in -2:2);
    all_shifts = cat(all_shifts..., dims=4);

    # pupil reconstruction model
    pupil_mask = (abs.(pupil_em) .> 0.1)
    pupil_mask[1] = false # fix the middle pixel to remove ambiguities
    modified_pupil = copy(pupil_noabber)
    # this forward model gets an object and complex-valued pupil values (only inside) and calculates ISM Data
    function fwd_amp(params; psf_ex=nothing)
        # its important to have the return value for the gradient to compute
        mod_pupil = into_mask!(params(:pupil_em), pupil_mask, modified_pupil)
        psf_em = abs2.(ifft(mod_pupil))
        # mod_ex_pupil = into_mask!(conj.(params(:pupil_em)), pupil_mask, modified_pupil)
        psf_ex = ifelse(isnothing(psf_ex), psf_em, psf_ex) # abs2.(ifft(mod_ex_pupil))
        all_psfs = psf_ex .* real.(ifft(fft(psf_em) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    pupil_strength = abs.(pupil_noabber[pupil_mask])
    # this forward model gets an object and pupil phase values (only inside) and calculates an ISM Data
    function fwd_phase(params; psf_ex=nothing)
        pupil_vals = pupil_strength .* cis.(params(:phase_em))
        mod_pupil = into_mask!(pupil_vals, pupil_mask, modified_pupil)
        psf_em = abs2.(ifft(mod_pupil))
        # mod_ex_pupil = into_mask!(conj.(pupil_vals), pupil_mask, modified_pupil)
        psf_ex = ifelse(isnothing(psf_ex), psf_em, psf_ex) # abs2.(ifft(mod_ex_pupil))
        all_psfs = psf_ex .* real.(ifft(fft(psf_em) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # intensity psf reconstruction model
    # this forward model gets an object and an intensity PSF and calculates the ISM data
    function fwd_int(params; psf_ex=nothing)
        psf_em = params(:psf_em)
        psf_ex = ifelse(isnothing(psf_ex), psf_em, psf_ex) # abs2.(ifft(mod_ex_pupil))
        all_psfs = psf_ex .* real.(ifft(fft(psf_em) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # make a simulation
    meas = sim_forward((x)->fwd_amp(x, psf_ex=psf_ex), (obj=Positive(obj), pupil_em=Fixed(pupil_em[pupil_mask])));
    meas = Noise.poisson(meas);

    # calculate some possible normalization factors.
    mymean = sum(meas) ./ prod(size(meas))
    mymax = maximum(meas) 
    numpix = prod(size(meas))

    # start with some object-only iterations:

    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean),  4.0*mymean), psf_em=Fixed(psf_noabber))
    # start_val = (obj=Normalize(Positive(res1[:obj]), 16.0*mymean), phase_em=Fixed(angle.(pupil_noabber[pupil_mask])))
    start_val = (obj=Normalize(Positive(ones(sz) .* mymean), 16.0*mymean), phase_em=Fixed(angle.(pupil_noabber[pupil_mask])))
    # res1, myloss1 = optimize_model(start_val, fwd_int, meas; iterations=50)
    get_loss(start_val, myfwd, meas)
    res1, myloss1 = optimize_model(start_val, fwd_phase, meas; iterations=50)
    print("Loss is $(myloss1[end])")
    plot(myloss1, label="L-BFGS") # , yaxis=:log

    @vt obj res1[:obj]

    recon_int = false
    phase_only = true
    myfwd = nothing
    if recon_int
        start_val = (obj=Normalize(Positive(res1[:obj]),  4.0*mymean), psf_em=Positive(psf_noabber))
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
    # just look at the initial loss
    get_loss(start_val, myfwd, meas)

    res, myloss = optimize_model(start_val, myfwd, meas; iterations=450)
    println("Loss is $(myloss[end])")
    plot!(myloss, label="L-BFGS") # , yaxis=:log
    if recon_int
        @vtp fftshift(psf_em) fftshift(res[:psf_em])
    else
        if phase_only
            into_mask!(pupil_strength .* cis.(res[:phase_em]), pupil_mask, modified_pupil)
        else
            into_mask!(res[:pupil_em], pupil_mask, modified_pupil)
        end
        @vtp fftshift(pupil_em) fftshift(modified_pupil)
    end



    # END here
    # here is some stuff how to do use the low-level interface instead:

    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(myfwd, start_val);
    # @vt forward(start_vals) meas

    @time first_pred = forward(start_vals)

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

end

