# do some fitting
using InverseModeling, Plots, DelimitedFiles, DeconvOptim, NDTools


d = @__DIR__ ;
fn1 = d * Base.Filesystem.path_separator * "IRF.dat";
irf = readdlm(fn1, '\t', Float64, '\n', skipstart=1, comment_char='\\');
fn2 = d * Base.Filesystem.path_separator * "FAD.dat";
dat = readdlm(fn2, '\t', Float64, '\n', skipstart=1, comment_char='\\');

fit_start = findfirst(dat[:,2] .> 1e4) - 100
irf_start = findfirst(irf[:,1] .> 1) - 100
nfit = size(dat[fit_start:end,1],1)
irf2 = irf[irf_start:irf_start+nfit-1] # irf_start+1+size(dat,1)

time_data = dat[fit_start:end,1] .* 0.004;
irf_n = circshift(irf2 ./ sum(irf2), (-107,));
plot(time_data,dat[fit_start:end,2])
plot!(time_data,dat[fit_start:end,3])
plot!(time_data, irf_n[:,1] .* 1000000)

function multi_exp(params)
    # tofit should contain: 
    t0 = params(:t0)
    offsets = params(:offsets)
    I_iso =  multi_exp_decay(time_data .- t0, params(:amps), params(:τs))
    r =  multi_exp_decay(time_data .- t0, params(:r0), params(:τ_rot))
    I_par = params(:crosstalk)[1].*soft_delta.(time_data.- t0)  .+ offsets[1] .+ (1 .+ 2 .* r) .* I_iso # .+ ref .* reflection.(t .- t0)
    I_perp = params(:crosstalk)[2].*soft_delta.(time_data .- t0) .+ offsets[2] .+ params(:G) .* (1 .- r) .* I_iso
    return cat(I_par, I_perp, dims=2)
end

# append the measured data along y to deal with all data simultaneously during the fit
measured = cat(dat[fit_start:end,2], dat[fit_start:end,3],dims=2)

norm_fac = size(measured,1) ./ sum(measured,dims=1)
measured_n = measured .* norm_fac # normalize the input data

otf, conv = plan_conv(measured_n, irf_n,(1,));

nf = dropdims(norm_fac,dims=1)
start_val = (t0=1.52, offsets=[100.0, 100.0].*nf, 
            τ_rot=[5.676, 2.0],
            r0=[0.2,0.2], G=[1.736169364281421],
            # amps=[2, 2, 2], τs =[0.5, 2.0,4.0],
            amps=[2, 2], τs =[2.0,4.0],
            crosstalk=[1000.0,500.0].*nf)
#            amps=[2.1, 0.5, 3.5], τs =[0.07,2.0,4.0], crosstalk=[1000.0,500.0].*nf)

#=
start_val = (amps=[2.1600984514146395, 3.5757088854321526], 
            τ_rot = [0.24565656618564113, 0.05583533631219779], 
            τs = [0.07844863401634174, 2.7657422569582137],
            crosstalk = [0.4417981316678886, 1.4701194213597888],
            G = [1.0685422015013588], t0 = 1.4928845790585399,
            offsets = [0.0630590091970162, 0.06839425119169348], 
            r0 = [0.2079541457576961, 0.20122977028271014])
=#

convolved = (vec) -> conv(multi_exp(vec));

start_vals, fixed_vals, forward, get_fit_results = create_forward(convolved, start_val)
plot(time_data,measured_n, xlabel="time (ns)", ylabel="intensity (a.u.)")
plot!(time_data,forward(start_vals))

optim_res = InverseModeling.optimize(loss(measured_n, forward), start_vals, iterations=800);
# @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
bare, res = get_fit_results(optim_res)
fit = forward(bare);

# print("t0: $(res[1])\noffsets: $(res[2])\ncrosstalk: $(res[3])\nτᵣₒₜ: $(res[4]) ns\nr₀: $(res[5])\nG: $(res[6])\namps: $(res[7])\nτs: $(res[8]) ns")

l = @layout([a; b]);
resid = measured_n .- fit;
toplot = cat(measured_n[:,1],resid[:,1],measured_n[:,2], resid[:,2], fit[:,1], resid[:,1], fit[:,2], dims=2);
labels = ["Iₚₐᵣ" "" "Iₚₑᵣₚ" "residₚₑᵣₚ" "fitₚₐᵣ" "residₚₐᵣ" "fitₚₑᵣₚ"]
plot_ref = plot(time_data,toplot, layout=l, title = ["Multiexponential Fit" "Residuals"], 
    xlabel = "time (ns)",
    ylabel = "intensity (a.u.)",
    label = labels)

start_vals = bare # reuse the result for next fit

savefig(plot_ref, "Anisotropy_fit_poisson.pdf")
