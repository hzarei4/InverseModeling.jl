using TimeTags, View5D, NDTools
using TestImages, MicroscopyTools
using Statistics
using InverseModeling

function main()
    q = read_flim(raw"D:\data\FLIM_data\LSM_15.pt3",  sx = 512, data_channel=2, marker_channel_y=18, is_bidirectional=false);
    # q2 = reshape(q[:,1:14080,:,:], (256,256,55,128));
    q = select_region(q,center=(253, 261), new_size=(100,100))
    new_axis = Colon();
    q = q[:,:,:,30:end, new_axis];
    @vt q
    @vv sum(q,dims=(3,4))
    @vv sum(q,dims=(1,2,3))

    mean_tau=size(q,4)/3.0
    mean_amp = mean(q)
    sz = (size(q)[1:3]..., 1, 2);
    # use individual dual exponentials with indidual lifetimes
    # tau_start = cat((mean_tau ./ 1.5f0) .*ones(Float32, sz[1:4]) , (mean_tau .* 1.5f0) .*ones(Float32, sz[1:4]) , dims=5);    

    # use two global lifetimes
    tau_start = cat((mean_tau ./ 1.5f0) , (mean_tau .* 1.5f0) , dims=5);

    # individual amps
    amp_start = mean_amp .*ones(Float32, sz);

    bare, res, fit = do_fit(q, amp_start, tau_start);
    @vt q fit (q .- fit)
    # @ve res[:amps] res[:τs] 
    @ve res[:amps] 
    res[:τs]

end

"""
    multi_exp_decay(time_data, amps, τs)

Create a multi exponential decay function in parallel for the whole 2D or 3D image.
The 4th dimension represents time and the 5th dimension the different decay components.
The parameters are individual for each pixel and supplied as a 2D or 3D array.

Arguments:
- `time_data`: The time data for the decay.
- `amps`: The amplitudes of the decay components. For multi-exponential decays, this can be multiple coefficients per pixel stacked along the `multi-dim` direction.
- `τs`: The decay time constants per pixel. For multi-exponential decays, this can be multiple coefficients per pixel stacked along the `multi-dim` direction.
- multi_dim: The dimension along which the decay components are summed up. Default is 5.
"""
# function  multi_exp_decay(time_data, amps, τs, multi_dim=5)
#     return sum(amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=multi_dim);
# end

"""
    single_exp(params)

Create a single exponential decay function in parallel for the whole 2D or 3D image.
The parameters are individual for each pixel and supplied as a 2D or 3D array.

    It is important that this function has minimal memory allocation
"""
function multi_exp(params, I_decay_mem, time_data = reorient(1:0.2f0:10, Val(4)))
    # tofit should contain: 
    t0 = params(:t0)
    offset = params(:offset)
    amps = collect(params(:amps))
    τs = collect(params(:τs))

    # the line below works, but is quite memory intensive
    # return sum(offset .+ amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs), dims=5);

    function accum!(mymem, n)
        mymem .+= offset .+ amps[:,:,:,:,n] .* MicroscopyTools.soft_theta_pw.(time_data, 0.002f0) .* exp.(.-time_data ./ τs[:,:,:,:,n])
    end
    # better version but there are some problems in the pullback ?
    return sum!_(I_decay_mem, accum!, size(amps,5));
end


"""
    test_lifetime_fitting()

performs a simulation and extracts the FLIM image.
"""
function test_lifetime_fitting()
    param_int1 = Float32.(testimage("resolution_test_512"))
    param_int1 = select_region(param_int1, center=(200,200), new_size=(200, 200))
    param_int2 = permutedims(param_int1,(2,1))
    param_int = cat(param_int1, param_int2, dims=5)
    sz = size(param_int)
    tau0 = 5.0f0
    param_tau = tau0 .+ 0.8f0 .*rand(Float32, sz...)
    fwd_vals = (t0=0.1f0, offset=0.1f0, amps=param_int, τs=param_tau)

    num_times=10
    times = reorient(0:0.2f0:num_times-1, Val(4))
    times = collect(times) # seems a little faster
    tmp_mem = zeros(Float32, sz[1:2]..., 1, size(times,4));
    fwd = (vec) -> multi_exp(vec, tmp_mem, times);

    ground_truth_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, fwd_vals, Float32);
    @time simulated = forward(ground_truth_vals);

    gradient(loss(measured_n, forward), ground_truth_vals)[1] # yields a ComponentVector

    # @vv simulated
    measured_n = simulated # .+ 0.1 .*randn(Float32, sz)
    mean_tau = mean(param_tau)
    mean_amp = mean(param_int)
    # use individual dual exponentials with indidual lifetimes
    tau_start = cat((mean_tau ./ 1.5f0) .*ones(Float32, sz[1:4]), (mean_tau .* 1.5f0) .*ones(Float32, sz[1:4]) , dims=5);    
    # individual amps
    amp_start = mean_amp .*ones(Float32, sz);
    
    bare, res, fit = do_fit(measured_n, mean_tau, amp_start; times=times[:], iterations=2);
    @vt measured_n fit (measured_n .- fit)
    ## lets try to invert this problem:
end

"""
    do_fit(measured_n, amp_start, tau_start; times=0:size(measured_n,4)-1)

performs the fit of `measured_n` with the starting amplitudes `amp_start` and the lifetimes `tau_start`.
"""
function do_fit(measured_n, amp_start, tau_start; times=0.2f0*(0f0:size(measured_n,4)-1), iterations=20)
    # convolved = (vec) -> conv(multi_exp(vec));
    all_start = (t0=0.12f0, offset=0.09f0, amps=Float32.(amp_start), τs=Float32.(tau_start))

    sz = size(measured_n)
    times = reorient(times[:], Val(4))
    tmp_mem = zeros(Float32, sz[1:2]..., 1, size(times,4));

    fwd = (vec) -> multi_exp(vec, tmp_mem, times);

    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, all_start);
    @time start_simulation = forward(start_vals);

    myloss = loss(measured_n, forward);

    # optim_res = InverseModeling.optimize(loss(measured_n, forward), start_vals, iterations=0);
    @time optim_res = optimize_model(myloss, start_vals, iterations=iterations);

    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res)
    fit = forward(bare);

    return bare, res, fit
end
