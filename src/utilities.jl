export get_loss
export into_mask!
export sum!_

"""
    @my_get t s

Access a NamedTuple/Struct `t` with the field `s`.
```
julia> t = (a=10, b=12.0)
(a = 10, b = 12.0)

julia> @my_get t a
10
```
"""
macro my_get(t, x)
    :($t.$x)
end


# function truncated_view(dat, dims)
#     return @view dat[(1:ifelse(d in dims, size(dat,d)-1, size(dat, d)) for d in ndims(dat))...]
# end

 # construct a named tuple from a dict
construct_named_tuple(d) = NamedTuple{Tuple(keys(d))}(values(d))

"""
    sum!_(mymem, accum!, N)

A quick and dirty sum! function which sets mymem first to zero and then calls the `accum!` function.
A total of N accumulations are executed in place on memory `mymem`.

This function was mainly written to be differentiable via an rrule.

# Arguments
+ `accum!` expects mymem as the first argument and an index `n` as the second.

"""
function sum!_(mymem, accum!, N)
    mymem .= zero(eltype(mymem))
    for n=1:N
        accum!(mymem, n)
    end
    return mymem
end

"""
    into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}

fills a vector into a mask `mymask`. This is very useful for optimizing only some pixels in an image. E.g. the inner part of a Fourier-space aperture.
#Arguments
+ `vec` : the vector to fill into the mask
+ `mymask`  : a binary mask to use
+ `tmp_data` : the mask is filled into only this part of the dataset. For efficiency reasons you should provide this argument, even if this is not strictly necessary as zeros are filled in.
            Note that you can also supply this argument to have a part of the resulting image being fixed.
#Returns
tmp_mask
"""
function into_mask!(vec::AbstractArray{T, 1}, mymask::AbstractArray{Bool, N}, tmp_data=zeros(eltype(vec),size(mymask))) where {T, N}
    tmp_data[mymask] = vec
    return tmp_data
end

# define custom adjoint for, since mutating arrays is not supported
function ChainRulesCore.rrule(::typeof(into_mask!), vec::AbstractArray{T, 1}, amask::AbstractArray{Bool, M},
    tmp_data=zeros(eltype(vec), size(amask))) where {T, M}
    Y = into_mask!(vec, amask, tmp_data)
    function into_mask_pullback(barx)
        if eltype(vec) <: Complex
            return NoTangent(), barx[amask], NoTangent(), NoTangent()
        else
            return NoTangent(), real(barx[amask]), NoTangent(), NoTangent()
        end
    end
    return Y, into_mask_pullback
end

function ChainRulesCore.rrule(::typeof(sum!_), mymem, accum!, N)
    Y = sum!_(mymem, accum!, N)
    accum_grad(a_mem, a_idx) = Zygote.pullback(accum!, a_mem, a_idx)[2];
    #val, mygrad = Zygote.withgradient(loss_fkt, vec)

    function sum!__pullback(y) # NOT YET CORRECT with the pullback of accum!, which is needed.
        # @show "sum!_ pullback"
        # @show size(mymem)
        # @show typeof(y)
        # sum!_(mymem, accum_grad, N)
        return NoTangent(), y .* mymem, mymem, NoTangent()
    end
    return Y, sum!__pullback
end
