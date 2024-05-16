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
function  multi_exp_decay(time_data, amps, τs, multi_dim=5)
    return sum(amps .* MicroscopyTools.soft_theta_pw.(time_data, 0.002) .* exp.(.-time_data ./ τs), dims=multi_dim);
end

"""
    single_exp(params)

Create a single exponential decay function in parallel for the whole 2D or 3D image.
The parameters are individual for each pixel and supplied as a 2D or 3D array.

"""
function multi_exp(params, time_data = reorient(1:0.2:10, Val(4)))
    # tofit should contain: 
    t0 = params(:t0)
    offset = params(:offset)
    # println("sizes: $(size(params(:amps))) $(size(params(:τs))) $(size(offset)) $(size(time_data)) $(size(t0))")
    I_decay = MicroscopyTools.soft_delta.(time_data.- t0)  .+ offset .+
            multi_exp_decay(time_data .- t0, params(:amps), params(:τs)) # .+ ref .* reflection.(t .- t0)
    return I_decay
end

function test_lifetime_fitting()
    param_int1 = Float32.(testimage("resolution_test_512"))
    param_int1 = select_region(param_int1, center=(200,200), new_size=(100,100))
    param_int2 = permutedims(param_int1,(2,1))
    param_int = cat(param_int1, param_int2, dims=5)
    sz = size(param_int)
    tau0 = 5.0f0
    param_tau = tau0 .+ 0.8f0 .*rand(Float32, sz...)
    fwd_vals = (t0=0.1f0, offset=0.1f0, amps=param_int, τs=param_tau)

    num_times=10
    fwd = (vec) -> multi_exp(vec, reorient(1:0.2:num_times, Val(4)));

    ground_truth_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, fwd_vals)

    @time simulated = forward(ground_truth_vals);
    @vv simulated
    measured_n = simulated # .+ 0.1 .*randn(Float32, sz)
    mean_tau = mean(param_tau)
    mean_amp = mean(param_int)
    # use individual dual exponentials with indidual lifetimes
    tau_start = cat((mean_tau ./ 1.5f0) .*ones(Float32, sz[1:4]) , (mean_tau .* 1.5f0) .*ones(Float32, sz[1:4]) , dims=5);    
    # individual amps
    amp_start = mean_amp .*ones(Float32, sz)
    
    bare, res, fit = do_fit(measured_n, mean_tau, mean_amp);
    @vt measured_n fit (measured_n .- fit)
    ## lets try to invert this problem:

end

function do_fit(measured_n,amp_start, tau_start; times=0:size(measured_n,4)-1)
    # convolved = (vec) -> conv(multi_exp(vec));
    all_start = (t0=0.12f0, offset=0.09f0, amps=amp_start, τs=tau_start)

    fwd = (vec) -> multi_exp(vec, reorient(times, Val(4)));

    start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, all_start)
    @time start_simulation = forward(start_vals);

    # optim_res = InverseModeling.optimize(loss(measured_n, forward), start_vals, iterations=0);
    @time optim_res = optimize_model(loss(measured_n, forward), start_vals, iterations=20);

    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res)
    fit = forward(bare);

    return bare, res, fit
end
