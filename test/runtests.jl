using Test

    # # do some Zygote tests
    # a = rand(32,32)
    # b = rand(32,32,1,25)
    # m = rand(32,32,1,25)
    # # all_psfs = psf_ex .* real.(ifft(fft(params(:psf_em)) .* all_shifts, (1,2,3)))
    # myloss(a) = sum(abs2.(a .* real.(ifft(fft(a) .* b, (1,2,3))) .- m))
    # @time gradient(myloss, a)[1];

        # gradient test
        a = rand(ComplexF32,10,10)
        b = rand(ComplexF32,10,10)
        m = abs.(a) .> 0.3
        v = a[m]
    
        lo(v1) = sum(abs2.(into_mask!(v1, m, b) .- 0.5))
        lo2(v1) = sum(abs2.(v1 .- 0.5))
        g = gradient(lo, v)[1]
        g2 = gradient(lo2, v)[1]
        g == g2
    
        #
        tst_val = (tst1=Normalize(Positive(2.0), 10.0),)
        tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
        forward(tst_start_vals)
    