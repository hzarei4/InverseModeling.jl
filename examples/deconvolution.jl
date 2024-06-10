# using InverseModeling, View5D, Noise, NDTools, FourierTools
using View5D, NDTools, FourierTools, TestImages
using IndexFunArrays, ComponentArrays
using InverseModeling
# using Plots
using Noise
using PointSpreadFunctions
using Statistics
# using SyntheticObjects
# using Zygote

function test_deconv()

    λ_em = 0.520
    sampling = (0.050,0.050,0.200)

    obj = select_region(Float32.(testimage("resolution_test_512")), new_size=(128,128), center=(249,374)) .* 50000;
    obj = Float32.(testimage("simple_3d_ball.tif"))

    sz = size(obj)

    # make a different (aberrated) psf
    # aberrations = Aberrations([Zernike_VerticalAstigmatism, Zernike_ObliqueAstigmatism],[0.5, 0.6])
    pp_em = PSFParams(λ_em, 1.4, 1.52, pol=pol_scalar, method=MethodPropagateIterative);
    # a_em = apsf(sz, pp_em; sampling=sampling);
    p_em = psf(sz, pp_em; sampling=sampling);
    otf = rfft(ifftshift(p_em))
    otf = otf ./ maximum(abs.(otf))

    # @vt p_ex p_em p_em2

    # intensity psf reconstruction model
    # this forward model gets an object and an intensity PSF and calculates the ISM data
    function fwd_conv(params)  # using the OTF as a clojure
        obj = params(:obj)  # not: use round brackets here. This is an overloaded access function
        return irfft(rfft(obj) .* otf, size(obj,1))
        # return conv(params(:obj), all_psfs, (1,2))
    end

    # make a simulation
    start_val = (obj=obj,)
    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_conv, start_val)

    pimg = forward(start_vals)
    nphotons = 100;
    nimg = poisson(pimg, nphotons)

    @vt obj pimg nimg

    # deconvolution
    start_val = (obj=Positive(mean(nimg).*ones(Float32, size(nimg))),)
    # start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd_conv, start_val)
    # optim_res = InverseModeling.optimize(loss(nimg, forward), start_vals, iterations=80);
    res1, myloss1 = optimize_model(start_val, fwd_conv, nimg; iterations=50)

    @vt obj nimg res1[:obj]

end

