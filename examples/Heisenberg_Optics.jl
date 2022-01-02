# fitting example to find out about the Heisenberg limit in optics

using InverseModeling, Plots, NDTools, IndexFunArrays, FourierTools
using ImageView, Optim, FFTW
sz = (128,128)

my_x2 = collect(abs2.(ramp(1, sz[1])))
my_y2 = collect(abs2.(ramp(2, sz[2])))
my_r3 = collect(abs.(rr(sz)) .* abs2.(rr(sz)))
function variance(psf)
    # (sum(psf.*my_x2) + sum(psf.*my_y2)) / sum(psf)
    sum(psf.*my_r3)/ sum(psf)
end

function stdx(psf)
    sqrt(sum(psf.*my_x2) / sum(psf))
end

R0 = sz[1]/8
ctr_mask = rr(sz) .< R0
mymask = ifftshift(ctr_mask) # shift mask to the corner
aperture = 1.0.*mymask

ppos = xx((sz[1],1))/R0

function to_pupil(avec)
    tmp_pupil = into_mask(avec, mymask)
    return tmp_pupil
end

to_ctr = fft(delta(sz))
function fwd(pupil)
    abs2.(ifft(to_ctr .* to_pupil(pupil))) # yields Matrix{Float64}
end

function loss(pupil)
    res= variance(fwd(pupil))
    println(res)
    return res
end

pupil = aperture[mymask]

loss(pupil)
plot(fwd(pupil))
start_val = (pupil=pupil,);

start_vals, fixed_vals, forward, backward, to_img = create_forward(fwd, start_val);

using Zygote
loss_fkt = x->loss(backward(x).pupil)
g!(G,vec) = G.=gradient(loss_fkt,vec)[1]
optim_options = Optim.Options(iterations=200)
@time optim_res = Optim.optimize(loss_fkt, g!, start_vals, LBFGS(), optim_options)

# optim_res = InverseModeling.optimize(loss_fkt, start_vals, iterations=10);
# @show optim_res = Optim.optimize(loss(measured, forward), start_vals, LBFGS())
bare, res = to_img(optim_res)
res_pupil = fftshift(to_pupil(res.pupil))
imshow(res_pupil)
res_fwd = fwd(res.pupil)
imshow(res_fwd)
res_pupil_ctr = res_pupil[:,sz[2]÷2]
ctr_pupil = ctr_mask[:,sz[2]÷2]
# plot(ppos,start_vals.pupil) # xlims=(-3,3)
plot(ppos,res_pupil_ctr ./ maximum(res_pupil_ctr))
plot!(ppos,ctr_pupil .* cos.(pi*ppos/2)) # theoretical solution

function heisenberg(pupil)
    print("Heisenberg Limit: Δx * Δp >= 1\n")
    Δx = stdx(abs2.(pupil))
    res_fwd = abs2.(ifft(to_ctr .* pupil))
    Δp = stdx(res_fwd)
    L = Δx * Δp
    print("Measured: $Δx * $Δp == $L\n")
end

my_gaussian = gaussian(sz,sigma=4.0)
imshow(my_gaussian)
imshow(abs2.(ifft(to_ctr .* my_gaussian)))
# ctr_gaussian = my_gaussian[:,sz[2]÷2]

heisenberg(my_gaussian)

heisenberg(res_pupil)

loss(start_vals.pupil)
loss(res.pupil)


savefig(plot_ref, "Heisenberg_fit.pdf")
