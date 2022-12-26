export loss_gaussian, loss_poisson, loss_anscombe, loss_poisson_pos, loss_anscombe_pos

loss_gaussian(data, fwd, bg=0) = sum(abs2.(data.-fwd))
loss_poisson(data, fwd, bg=eltype(data)(0)) = sum((fwd.+bg) .- (data.+bg).*log.(fwd.+bg))
loss_poisson_pos(data, fwd, bg=eltype(data)(0)) = sum(max.(eltype(fwd)(0),fwd.+bg) .- max.(eltype(fwd)(0),data.+bg).*log.(max.(eltype(fwd)(0),fwd.+bg)))
loss_anscombe(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(data.+bg) .- sqrt.(fwd.+bg)))
loss_anscombe_pos(data, fwd, bg=eltype(data)(0)) = sum(abs2.(sqrt.(max.(eltype(fwd)(0),data.+bg)) .- sqrt.(max.(eltype(fwd)(0),fwd.+bg))))

