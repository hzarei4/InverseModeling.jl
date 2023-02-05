export gauss_model, gauss_start, gauss_fit

function gauss_model(sz, parameters)
    pos = collect(idx(sz))
    function agauss(pos, i0, x0, σ, offset)
        # tbuf = sum(abs2.((pos .- x0)./σ)) # cannot be used 
        tbuf = abs2.((pos[1].-x0[1]) ./σ[1]) #sum(abs2.((pos .- x0)./σ))
        for n=2:lastindex(x0) # length(sz)
            tbuf += abs2.((pos[n].-x0[n]) ./σ[((n-1)%length(σ))+1])   #sum(abs2.((pos .- x0)./σ))
        end
        if length(σ) > length(x0) # include covariance terms
            c=1
            for n=2:length(x0)÷2+1 # iterate over covariance terms (upper triangle in covariance matrix)
                for m=1:n-1 # iterate over covariance terms
                    tbuf += (pos[n].-x0[n]).*(pos[m].-x0[m]) .* σ[length(x0)+c]
                    # @show σ[length(x0)+c]
                    c += 1
                end
            end
        end
        return offset + i0.*exp(.-tbuf./2)
    end
    # using Ref() protects these vector-type arguments from immediate broadcasting here
    # buffer to store the temporary calculation
    tmp = zeros(sz)
    function fit_fkt(params)
            tmp .= agauss.(pos, params(:i0), Ref(params(:μ)), Ref(params(:σ)), params(:offset)) 
        return tmp
    end

    fit_parameters, fixed_vals, forward, backward, get_fit_results = create_forward(fit_fkt, parameters)
    return forward, fit_parameters, get_fit_results
end

function tuple_sum(tarray)
    tsum(t1,t2) = t1 .+ t2
    reduce(tsum, tarray) 
end

function gauss_start(meas, rel_thresh=0.2; has_covariance=false)
    pos = idx(size(meas))
    offset = minimum(meas)
    i0 = maximum(meas) - offset
    meas = meas .- offset .- i0 .* rel_thresh
    mymask = meas .> 0
    sumpix = sum(meas .* mymask)
    tosum(apos, dat) = apos .* dat 
    μ = tuple_sum(tosum.(pos, meas.*mymask)) ./ sumpix
    mysqr(apos, dat) = abs2.(apos) .* dat 
    pos = idx(size(meas), offset=size(pos).÷2 .+1 .+ μ)
    σ = 1.2 .* max.(1.0,sqrt.(tuple_sum(mysqr.(pos, meas.*mymask )) ./ sumpix))
    if has_covariance
        σ = [σ..., zeros(((length(μ))*(length(μ)-1))÷2)...]
        @show σ
    end
    return (i0 = i0, σ=[σ...], μ=[μ...], offset=offset) # Fixed(), Positive
end

"""
    gauss_fit(meas, start_params=(), ndims=[]; verbose=false, pixelsize=1.0, optimizer=LBFGS(), iterations=100, noise_model=gaussian_norm)

performs a fit of an ND-Gaussian function to ND data. 
#arguments
+ `meas`    : the (measured) data to fit the Gaussian to
+ `start_params` : if empty, the start parameters will be estimated based on center of mass and variance calculations.
            Alternatively, a named tuple `(:i0,:μ,:σ,:offset)` can be provided, containing
            `:offset` the offset
            `:i0` the maximum value over the offset
            `:μ` a vector of mean positions
            `:σ` a vector of variances along the X and Y coordinates. Note that rotated Gaussians with a covariance matrix are currently not supported.
            Note that any of the parameters can be fixed by encompassing them with the modifyer `Fixed()`
+ `ndims` : currently not used
+ `verbose`: if true, information on the fit will be printed on screen.
+ `pixelsize`: a scalar or tuple that the resulting standarddeviation and FWHM will be multiplied with
+ `optimizer`: the optimization method to use. See the `Optim` toolbox for details. `NelderMead()` and `LBFGS()` are possible choices.
+ `iterations`: maximum number of iterations
+ `noise_model`: defines the norm to optimize. default is gaussian_norm, but also other norms can be used
+ `bg`: an optional background term which is included in some of the noise models (e.g. `loss_poisson` and `loss_anscombe`)
#returns
a named tuple with the result parameters as a tuple (see above), additionally containing the parameter `:FWHM` for convenience referring to the Full width at half maximum of the Gaussian
and the fit forward projection and the result of the optim call.
"""
function gauss_fit(meas, start_params=[], ndims=[]; verbose=false, pixelsize=1.0, optimizer=LBFGS(), iterations=100, noise_model=loss_gaussian, bg=eltype(meas)(0), has_covariance=false)
    start_params = let
        if isempty(start_params)
            gauss_start(meas, has_covariance=has_covariance)
        else
            start_params            
        end
    end
    # @show start_params
    scale = get_val(start_params[:i0])
    # meas = meas ./ scale
    start_params = (i0 = Normalize(start_params[:i0], scale), σ=start_params[:σ], μ=start_params[:μ], offset=Normalize(start_params[:offset], scale))
    forward, fit_parameters, get_fit_results = gauss_model(size(meas), start_params)
    # res = Optim.optimize(loss(meas, forward), fit_parameters)
    res = optimize_model(loss(meas, forward, noise_model, bg), fit_parameters, optimizer=optimizer, iterations=iterations)
    # fit_params = Optim.minimizer(res)
    bare, fit_params = get_fit_results(res)
    #fit_params[:σ] .*= pixelsize
    #fit_params[:i0] = fit_params[:i0] * scale
    #fit_params[:offset] = fit_params[:offset] * scale
    FWHMs = get_val(fit_params[:σ]) .* sqrt(log(2) *2)*2
    my_fwd = forward(bare) # .* scale
    # meas .*= scale
    mean_meas = sum(meas)/prod(size(meas))
    R2 = 1.0 - (sum(abs2.(meas .- my_fwd)) / sum(abs2.(meas .- mean_meas))) #ssq_res / ssq_total
    fit_params=merge(fit_params, (FWHMs=FWHMs,))
    # fit_params = (i0 = scale*fit_params[:i0], σ=fit_params[:σ], μ=fit_params[:μ], offset=fit_params[:offset]*scale, FWHM=FWHMs, R2=R2)
    if verbose
        println("FWHMs : $FWHMs, Iterations : $(res.iterations), Norm: $(res.minimum)")
    end
    return fit_params, my_fwd, res
end

