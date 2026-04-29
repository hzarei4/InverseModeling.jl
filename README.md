# InverseModeling.jl
is a general purpose toolbox aiming to streamline the construction and execution of inverse models.
In other words, it helps fitting various model parameters to experimental data using statistical models, regularizers and contraints.
The optimization itself mostly uses grandient-based optimizers, present in the `Optim.jl` toolbox.
The general framework is represented by this graph:

```
              -->- Pre-Forward    --<- apply gradient --<- ---<--- Regularizer
             |               |                           |
  Inverse Pre-Forward     Pre Forward                Adjunct Pre
             |               |                           |
Start-Values +  -<- Result -<-|                           |
                         Forward                     Adjunct
                             |                           |
                             |                           |
                             -->- compare (Noise Model) --
```
A speciality in this framework is the introduction of the `Pre-Forward-Model` which can serve to enforce domain-contraints to restrict the possible values or the dimensionality of the search-space.

Here is an example of a typical inverse model of a two-dimensional parabolic function to be fitted to data, which is also simulated:
```julia
using InverseModeling
times = -10:10
fit_fkt(params)::Vector{Float64} = (times.*params(:σ)[1]).^2 + (times.*params(:σ)[2]).^3 .+ times .* params(:μ)[1]
start_val = (σ=Positive([8.0, 25.0]), μ=Fixed([1.0, 3.0]) )
```
The `fit_fkt` is the actual model. Of course in real-life situations this function is typically complicated. However, it needs to be handles by automatic differentiation.
Note that in the forward model the model parameters, as visibl in `start_val` have to be accessed in a rather unusual way: as function calls to the input (here `params`) with names (e.g. : :σ). This allows the framework to process the parameters correctly.
So the parameter `params(:σ)[1]` refers to the starting value `25.0`.
Starting values are decorated with one or multiple properties, which are part of th Pre-Forward model. Here `Positive()` guarantees that this parameter (or array of parameters) is constrained to be all-positive in values.
The pre-forward decorator `Fixed` specifies that this parameter should not be part of the model, but remain constant during iterations. This is a convenient way to play with the model without needing significant alterations to
the actual code.
```julia
start_vals, fixed_vals, forward, backward, get_fit_results = create_forward(fit_fkt, start_val)
```
The function `create_forward` packages the model in a convenient way for later use. Here `start_vals` refer the the actual starting values provided to the iterative minimization algorithm. They are obtained by applying the inverse of the Pre-Forward model to the provided `stat_val`.
For convenience, all `fixed_vals` as specified by the keyword `Fixed` are returned here as a named tuple. The return value `forward` contains the full forward model to be applied to the `start_vals` (see below). The function `get_fit_results` is a convenient way to obtain the fit results.
```julia
meas = forward(start_vals)
meas .+= 5e5*randn(size(meas))
# distort the ground truth parameters a little
start_vals.σ += [0.5,0.2]
```
Here a perfect measurement `meas` result is created by applying `forward` to the `start_vals` (which are transformed `start_val` parameters.
Then noise is added to the `meas` variable and also the start_vals are distorted to challenge the fit.

```julia
@show optim_res = optimize_model(loss(meas, forward), start_vals)
```
This line optimizes (i.e. fits the data) using a `loss`, which can also specify the noise model used. Here a fixed Gaussian noise model is used, since no specific noise model was provided.
Optionally parameters, such as the choice of method or number of iterations can be forwarded to the `Optim.jl` toolbox.
```julia
bare, fit_res = get_fit_results(optim_res)
fit = forward(bare)

using Plots
plot(times, forward(start_vals), label="start values")
scatter!(times, meas, label="measurement")
plot!(times, fit, label="fit")
```
The previously returned `get_fit_results` function allows to conveniently extract the fit result parameters (including the fixed ones).
The fit can simply be obtained by running the forward model on the `bare` fit results.

Here is a list of supported functionalities:
## Pre-Forward Model
+ Fixed: Do not fit these parameters
+ Positive: Enforce positivity using a sqr function
+ PositiveH: Enforce positivity using a monotonic piecewise hyperbolic-linear function
+ Normalize: Allows various ways to normalize the fit parameters
+ ClampSum: Fixes the sum of all parameters to a constant and fitting in a reduced number of dimensions
+ BoundedSoftMax: Constrains the fit range to a lower and upper bound using a ratioed exponential
+ BoundedSoftMaxH: Constrains the fit range to a lower and upper bound using a ratioed hyperbolic piecwise linear function.

## Noise Models
+ loss_gaussian:  Gaussian noise model with a fixed variance. 
+ loss_poisson: Poisson noise model where the variance is equal to the mean. An additional background is allowed
+ loss_poisson_pos: Poisson noise model where the variance is equal to the mean. An additional background is allowed and positivitiy of the prediction is enforced.
+ loss_anscombe: An Anscombe transform is applyed to the data as well as the forward model. This performs similar to the Poisson model but typically converges faster.
+ loss_anscombe_pos: This version additionally enforces positivity.

## Inbuilt Models
Currently Gaussian fitting is provided via an inbuilt model.
