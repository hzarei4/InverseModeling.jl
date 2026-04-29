@testset "Modifyer Normalize" begin
    myval = 2.0
    mynorm = 3.0
    tst_val = (tst1=Normalize(myval, mynorm), )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    @test isapprox(forward(tst_start_vals), myval)
    @test isapprox(tst_start_vals.tst1, myval/mynorm)

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, myval)
end

@testset "Modifyer Fixed" begin
    tst_val = (tst1=10.0, tst2=Fixed(Normalize(Positive(15.0), 10)))
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    @test fixed_vals == InverseModeling.ComponentVector((tst2=15.0,))
end

@testset "Modifyer ClampSum" begin
    val = rand(3,4)
    tst_val = (tst1=ClampSum(val), )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test size(tst_start_vals.tst1) == (prod(size(val))-1,)
    @test tst_start_vals.tst1[:] == val[1:end-1]
end

@testset "Modifyer Positive" begin
    val =15.0
    tst_val = (tst1=Positive(val), )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test tst_start_vals.tst1 > 0
    @test tst_start_vals.tst1 ≈ sqrt(val)
end

@testset "Modifyer BoundedSoftMax" begin
    val = [0.5, 0.2, 0.9]
    tst_val = (tst1=BoundedSoftMax(val, 0, 1),)
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test tst_start_vals.tst1[1] ≈ 0
end

@testset "Modifyer PositiveH" begin
    val =15.0
    tst_val = (tst1=PositiveH(val), )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test tst_start_vals.tst1 > 0
    @test tst_start_vals.tst1 ≈ 14

    val = 0.001
    tst_val = (tst1=PositiveH(val), )
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test tst_start_vals.tst1 ≈ -999
end

@testset "Modifyer BoundedSoftMaxH" begin
    val = [5f0, 0.2f0, 10f0]
    tst_val = (tst1=BoundedSoftMaxH(val, 0, 20),)
    tst_start_vals, fixed_vals, forward, backward, get_fit_results = create_forward((x)->x(:tst1), tst_val);
    @test eltype(tst_start_vals.tst1) == Float32

    fwd = forward(tst_start_vals)
    @test isapprox(fwd, val)
    @test tst_start_vals.tst1[3] ≈ 0
end

