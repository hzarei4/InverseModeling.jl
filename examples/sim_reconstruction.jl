using InverseModeling, View5D, Noise, TestImages, FourierTools, FFTW

# parameters
# is a named Tuple with the following entries:
# 


# function sim_model(sz, parameters)
# returns a forward model, a set of fit parameters and a function to retrieve the fit results, 
# all of which can be used for fitting structured illumination microscopy data
# arguments:
# sz: size of the resulting array
# parameters: a number of parameters defining the SIM model.
function sim_model(sz, parameters)
    pos = collect(idx(sz))
    function fwd(pos, i0, x0, σ, offset)
        # tbuf = sum(abs2.((pos .- x0)./σ)) # cannot be used 
        tbuf = abs2.((pos[1].-x0[1]) ./σ[1]) #sum(abs2.((pos .- x0)./σ))
        for n=2:length(x0) # length(sz)
            tbuf += abs2.((pos[n].-x0[n]) ./σ[n])   #sum(abs2.((pos .- x0)./σ))
        end
        return offset + i0.*exp(.-tbuf./2)
    end
    # using Ref() protects these vector-type arguments from immediate broadcasting here
    # buffer to store the temporary caculation
    tmp = zeros(sz)
    function fit_fkt(params)
        tmp .= fwd.(pos, params(:i0), Ref(params(:μ)), Ref(params(:σ)), params(:offset)) 
        return tmp
    end

    fit_parameters, fixed_vals, forward, backward, get_fit_results = create_forward(fit_fkt, parameters)
    return forward, fit_parameters, get_fit_results
end


# calculates the intensity order strength matrix from a list of amplitude k-vectors and strength
# Parameters
# ----------
# SIMParam : Selection of SIM parameters
# Returns
# -------
# tuple with matrix of intensity k-vectors and matrix of order strengths
# See also
# -------
# Example
# -------
function mkIntKGrid(k_coherent_full, kall, IncoherentOrderstrengths, astrengths_full)  # ,kmax=None):
    ks = np.array(k_coherent_full)  # ks,astrengths
    nks = len(ks)
    kall = [ks[0] * 0.0]  # initializes the kall calculation
    IncoherentOrderstrengths = [0.0]
    ErrorLimit = 1.0e-6;
    astrengths = astrengths_full
    SIMParam.doesMatter = zeros(Bool, (nks, nks));
    if astrengths.ndim > 1:
        astrengths = transpose(astrengths)
    end
    for n in range(nks):
        for m in range(n, nks):  # only generate one side of the k-vectors
            dk = ks[m] - ks[n]
            #            print("Interfering ",n,"x",m)
            SIMParam.doesMatter[n, m] = (np.dot(astrengths[n], astrengths[m]) > 0.0)  # at least one instance in time, where both of these are used.
            if m != n and SIMParam.doesMatter[n, m] and np.linalg.norm(dk) < ErrorLimit:
                raise ValueError("Found two times the same k-vector (" + str(n) + "," + str(m) + "). This is against the rules!")
            end
            if SIMParam.doesMatter[n, m]:  # and (kmax is None or np.abs(dk) < kmax):
                if m != n:
                    SIMParam.kall.append(dk)
                    SIMParam.IncoherentOrderstrengths.append(astrengths[n] * astrengths[m] * 2.0)  # the 2.0 is for the other combination
                else:  # add to the zero order
                    SIMParam.IncoherentOrderstrengths[0] += astrengths[n] * astrengths[m]
                end
            end
    SIMParam.kall = np.array(SIMParam.kall)
    SIMParam.IncoherentOrderstrengths = np.transpose(np.array(SIMParam.IncoherentOrderstrengths))
    return SIMParam
end

# performs a non-tensorflow SIM simulation based on the parameters in SimParam.
# sim_param : structre of simulation parameters
function simulate_SIM(obj=nothing; use_2D=true, psf=nothing, PSF_params=nothing, pixelsize=(30, 30), downsamplefactors=nothing)
    if use_2D
        if isnothing(obj)
            obj = Float32.(testimage("resolution_test_512.tif"))
        end
        if isnothing(psf) 
            # objds=nip.resample(SimParam.obj,1.0/SimParam.downsamplefactors)
            psf = psf2d(size(obj), PSF_params=nothing)  # psf needs to be in the final nimg pixelsize due to downsampleConvolve
        end
    else
        if isnothing(obj)
            obj = Float32.(testimage("simple_3d_ball.tif"))
        end
        if isnothing(psf)
            psf = Float32.(testimage("simple_3d_psf.tif"))  # part of the nip toolbox
        end
    end

    psfd = let 
        if !isnothing(downsamplefactors)
            resample(psf, 1.0 ./ downsamplefactors)
        else
            psf
        end
    end
    rotf = rfft(ifftshift(psfd)) # converts the PSF to a half-complex rotf

    if !isnothing(k_coherent) isnothing(sim_paramnt_full)
        sim_param = rotate_SIM_coherent(SimParam)  # coherent k-vectors in all directions
    end

    SimParam = im.mkIntKGrid(sim_param)  # intensity k-vectors

    SimParam = im.genSIMConfiguration(SimParam)  # constructs the movement vectors and kall and the orderstrength
    SimParam.myillu = im.genSIMPattern(SimParam.obj, SimParam)
    # v5(illu)

    # m = 1.0
    # illu = (1 + m*im.coswave(obj,k0))

    if SimParam.NumPhot is not None:
        normed_obj = SimParam.obj / np.max(SimParam.obj) * SimParam.NumPhot  # account for the maximally emitted photons
        SimParam.obj = normed_obj

    if SimParam.use_bleaching:
        SimParam.emission = im.ApplyIlluWithBleaching(SimParam.obj, SimParam.myillu, SimParam.bleachingrate)
    else:
        SimParam.emission = SimParam.obj * SimParam.myillu

    SimParam.pimg = nip.downsampleConvolveROTF(SimParam.emission, SimParam.rotf, psfd.shape)
    #    SimParam.pimg = nip.convROTF(SimParam.emission,SimParam.rotf)
    SimParam.pimg[SimParam.pimg < 0.0] = 0.0

    if SimParam.ApplyNoise:
        SimParam.nimg = nip.noise.poisson(SimParam.pimg, seed=0, dtype='float32')  # Christian: default should be seed=0
    else:
        SimParam.nimg = SimParam.pimg
    SimParam.nimg = SimParam.nimg + SimParam.backgroundOffset

    return SimParam
end

function defaultReconParam(free_amplitudes=false)  # a structure for the reconstruction parameters
    # apply noise to initialization parameters.
    (perturb_object = false,  # If False, the perfect parameters are used (including the object)
    perturb_k_coherent = 0,  # 0.001
    perturb_k_ampstrength = 0,  # 0.2
    perturb_zeroPos = 0,  # 10.0
    perturb_bleach = false,

    downsamplefactors = [2.0, 2.0],
    ForcePos = true,  # Force object to be positive
    FreeAmplitudes = false,  # fit each amplitude (exept the first). If False: Fit only one set of amplitudes
    determineOffset = false,
    backgroundOffset = 0.0,
    useZeroPos = false,  # assume the the zero positions are fixed
    toOptimize = (Object=true,
                  k_coherent= true,
                  k_ampstrength= true,
                  zeroPos = true,
                  k_bleach = false,
                  k_dirstrength= ! free_amplitudes),
    borderRegion = [0, 0])
end


function test_reconstruction()
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
    @ve meas, res_img
end

