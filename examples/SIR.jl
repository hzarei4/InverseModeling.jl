# an example demostrating the use of this package for fitting a simple SIR model
# noisy SIR data is simulated and then fitted
using InverseModeling, Plots, Noise

function fwd(params)
    # Note that you need to use round brackets here, not square brackets
    infected = params(:infected)
    recovered = params(:recovered)
    infection_rate = params(:infection_rate)
    recovery_rate = params(:recovery_rate)
    all_inf = Float64[]
    timepoints = 100
    for t=1:timepoints
        push!(all_inf, infected)
        susceptible = 1.0 - infected - recovered
        new_recovered = recovery_rate * infected
        infected += infected * susceptible * infection_rate
        recovered += new_recovered
        infected -= new_recovered
    end
    all_inf
end


function main()
    # true_val = (infected = 0.01, recovered= 0.0, infection_rate=0.3, recovery_rate=0.1) # Fixed()
    # m = fwd(true_val)
    # plot(m)

    true_val = (infected = Fixed(0.02), recovered= Fixed(0.0), infection_rate=0.3, recovery_rate=0.1) # Fixed()
    fit_parameters, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, true_val)
    m = forward(fit_parameters)
    plot(m, label="ground truth")
    noisy = add_gauss(m, 0.03)
    plot!(noisy, label="noisy")

    start_vals = (infected = Fixed(0.02), recovered= Fixed(0.0), infection_rate=Positive(Fixed(0.3)), recovery_rate=0.05) # Fixed()
    start_fit, fixed_vals, forward, backward, get_fit_results = create_forward(fwd, start_vals)
    optim_res = InverseModeling.optimize(loss(noisy, forward), start_fit, iterations=100);
    # @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
    bare, res = get_fit_results(optim_res)
    @show res # show the result
    fit = forward(bare);
    plot!(fit, label="fit")

end


