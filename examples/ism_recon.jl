# using InverseModeling, View5D, Noise, NDTools, FourierTools
using View5D, NDTools, FourierTools, TestImages, IndexFunArrays, ComponentArrays
using InverseModeling
using Plots
using Zygote, Noise
using PointSpreadFunctions

function test_ism_recon()

    obj = select_region(Float32.(testimage("resolution_test_512")), new_size=(128,128), center=(249,374)) .* 1000;
    sz = size(obj)
    # pupil = disc(sz, sz[1]/4)
    # psf_ex = abs2.(ift(pupil))
    # psf_ex = ifftshift(psf_ex ./ sum(psf_ex))
    pp_ex = PSFParams(0.488, 1.4, 1.52, pol=pol_scalar);
    p_ex = psf(sz, pp_ex; sampling=(0.050,0.050,0.200));

    # make a different (aberrated) psf
    # pupil_em = disc(sz, sz[1]/5)
    # psf_em = abs2.(ift(pupil_em))
    # psf_em = ifftshift(psf_em ./ sum(psf_em))
    aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[1.5, 0.6])
    pp_em = PSFParams(0.520, 1.4, 1.52, pol=pol_scalar, method=MethodPropagateIterative ,aberrations=aberrations);
    a_em = apsf(sz, pp_em; sampling=(0.050,0.050,0.200));
    @vp ft(a_em)
    p_em = psf(sz, pp_em; sampling=(0.050,0.050,0.200));

    @vt p_ex p_em
    
    psf_ex = ifftshift(p_ex)
    psf_em = ifftshift(p_em)

    dx = 1.1; dy = 1.1;
    all_shifts = Tuple(ifftshift(exp_ikx(sz,shift_by=(x*dx,y*dy))) for x in -2:2, y in -2:2);
    all_shifts = cat(all_shifts..., dims=4);

    function fwd(params)
        all_psfs = psf_ex .* real.(ifft(fft(params(:psf_em)) .* all_shifts, (1,2,3)))
        return real.(ifft(fft(params(:obj)) .* fft(all_psfs, (1,2,3)), (1,2,3)))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # gt_params = (obj=obj, psf_em=psf_em)

    # meas = poisson(fwd(gt_params))
    # meas = poisson(fwd(gt_params))

    # make a simulation
    gt_val = (obj=Positive(obj), psf_em=Fixed(psf_em))
    gt_vals, fixed_vals, gt_forward, backward, get_fit_results = create_forward(fwd, gt_val)
    meas = Noise.poisson(gt_forward(gt_vals));
    size(meas)

    mymean = sum(meas) ./ prod(size(meas))
    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Positive(psf_ex))
    # psfmean = sum(psf_ex) ./ psf_ex
    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Normalize(Positive(psf_ex), psfmean))
    numpix = prod(size(meas))
    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), numpix), psf_em=ClampSum(Positive(psf_ex)))
    start_val = (obj=Normalize(Positive(ones(sz) .* mymean), numpix), psf_em=Positive(psf_ex))
    # start_val = (obj=Normalize(Positive(ones(sz) .* mymean), mymean), psf_em=Fixed(psf_em))
    # start_val = (obj=Positive(ones(sz) .* mymean), psf_em=Fixed(psf_em))
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, start_val)
    size(forward(start_vals)) 
    # @vt forward(start_vals) meas

    @time forward(start_vals);

    # # do some Zygote tests
    # a = rand(32,32)
    # b = rand(32,32,1,25)
    # m = rand(32,32,1,25)
    # # all_psfs = psf_ex .* real.(ifft(fft(params(:psf_em)) .* all_shifts, (1,2,3)))
    # myloss(a) = sum(abs2.(a .* real.(ifft(fft(a) .* b, (1,2,3))) .- m))
    # @time gradient(myloss, a)[1];

    loss(meas, forward)(start_vals)
    @time gradient(loss(meas, forward), start_vals)[1];


    @time optim_res = InverseModeling.optimize_model(loss(meas, forward, loss_gaussian, 0.01), start_vals, iterations=55);
    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    print("Loss is $(optim_res.minimum)")
    bare, res = get_fit_results(optim_res)

    # @ve obj, res[:obj]
    @vt fftshift(psf_ex) fftshift(psf_em) fftshift(res[:psf_em])
    plot(psf_ex[1:10,1], label="psf_ex")
    plot!(psf_em[1:10,1], label="GT psf_em")
    plot!(res[:psf_em][1:10,1], label="recovered psf_em")
end

