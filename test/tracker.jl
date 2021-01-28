using Tracker, Test, NNlib
using Tracker: TrackedReal, gradient, gradcheck, grad, checkpoint, forwarddiff
using NNlib: conv, ∇conv_data, depthwiseconv
using PDMats
using Printf: @sprintf
using LinearAlgebra: diagm, dot, LowerTriangular, norm, det, logdet, logabsdet, I, Diagonal
using Statistics: mean, std
using Random
# using StatsBase

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)
@testset "Tracker" begin
@test gradtest((x, W, b) -> σ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W) -> σ.(W*x), 5, (2,5))
@test gradtest((x, W, b) -> σ.(W*x .+ b), (5,3), (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), 5, (2,5), 2)
@test gradtest((x, W, b) -> logσ.(W*x .+ b), (5,3), (2,5), 2)
@test gradtest((w, x) -> w'*x, randn(Float64,10, 2), randn(Float64,10))
@test gradtest((w, x) -> w*x', randn(Float64,5,5), randn(Float64,5,5))
let A=Diagonal(randn(5, 5))
    @test gradtest(x-> A * x, (5,))
    @test gradtest(x-> A * x, (5, 5))
    @test gradtest(x-> x * A, (5, 5))
end
@test gradtest(x -> sum(x, dims = (2, 3)), (3,4,5))
@test gradtest(x -> sum(x, dims = 1), randn(Float64,2,3))
@test gradtest(x -> sum(x, dims = [1,2]), randn(Float64,2,3))
@test gradtest(x -> sum(x), randn(Float64,2,3))
@test gradtest(x -> prod(x, dims=(2, 3)), (3,4,5))
@test gradtest(x -> prod(x), (3,4,5))

@test gradtest(x -> softmax(x; dims = 1).*(1:3), 3)
@test gradtest(x -> softmax(x; dims = 1).*(1:3), (3,5))
@test gradtest(x -> softmax(x; dims = 2).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x; dims = 1).*(1:3), 3)
@test gradtest(x -> logsoftmax(x; dims = 1).*(1:3), (3,5))
@test gradtest(x -> logsoftmax(x; dims = 2).*(1:3), (3,5))

@test gradtest(x -> x', rand(5))

@test gradtest(det, (4, 4))
@test gradtest(logdet, map((x) -> x*x', (rand(4, 4),))[1])
@test gradtest((x) -> logabsdet(x)[1], (4, 4))

@testset "indexing & slicing" begin
  gradtest(x->view(x, 1:2, 1:2), rand(4, 4))
end

function promotiontest(f, A, B, C)
  r0 = f(A, B, C)
  r1 = f(param(A), B, C)
  r2 = f(A, param(B), C)
  r3 = f(A, B, param(C))
  r4 = f(param(A), param(B), param(C))

  @test !isa(r0, TrackedArray)
  @test all(isa.([r1,r2,r3,r4], TrackedArray))
  @test r1 == r2 == r3 == r4
  @test r0 == Tracker.data(r4)
end

@testset "concat" begin
  cat1(x...) = cat(x..., dims = 1)
  cat2(x...) = cat(x..., dims = 2)
  rvcat(x...) = reduce(vcat, x)
  rhcat(x...) = reduce(hcat, x)

  @testset for vcatf in [vcat, cat1, rvcat]
    @test gradtest(vcatf, rand(5), rand(3))
    @test gradtest(vcatf, rand(5), rand(3), rand(8))
    @test gradtest(vcatf, rand(5)', rand(5)')
    @test gradtest(vcatf, rand(5,2), rand(3,2), rand(8,2))
    @test gradtest(vcatf, rand(5,2,3), rand(3,2,3), rand(8,2,3))
    @test gradtest(vcatf, rand(5), rand(3,1))
    @test gradtest(vcatf, rand(5)', rand(2,5))
  end


  @testset for hcatf in [hcat, cat2, rhcat]
    @test gradtest(hcatf, rand(5), rand(5))
    @test gradtest(hcatf, rand(5)', rand(5)')
    @test gradtest(hcatf, rand(2,5), rand(2,3), rand(2,8))
    @test gradtest(hcatf, rand(2,5,3), rand(2,3,3), rand(2,8,3))
    @test gradtest(hcatf, rand(5), rand(5), rand(5,2))
    @test gradtest(hcatf, rand(5)', rand(1,3))
    @test gradtest(hcatf, rand(5), rand(5,2))
end

  @testset for catf in [vcat, cat1, rvcat, hcat, cat2, rhcat, (x...) -> cat(x..., dims = 3), (x...) -> cat(x..., dims = (1,2))]
    @test gradtest(catf, rand(5))
    @test gradtest(catf, rand(5)')
    @test gradtest(catf, rand(2,5))
    @test gradtest(catf, rand(2,5,3))
  end

  @test gradtest((x...) -> cat(x..., dims = 3), rand(2,5,2), rand(2,5,3), rand(2,5,4))

  @testset "cat($dim, ...)" for dim in 3:5
    catdim = (x...) -> cat(x..., dims = dim)
    @test gradtest(catdim, rand(5), rand(5), rand(5))
    @test gradtest(catdim, rand(2,5), rand(2,5), rand(2,5))
    @test gradtest(catdim, rand(2,5,3), rand(2,5,3), rand(2,5,3))
    dim == 3 && @test gradtest(catdim, rand(2,5,3), rand(2,5,0), rand(2,5,3))
  end

  @test !isa(vcat(rand(2)), TrackedArray)
  @test !isa(hcat(rand(2)), TrackedArray)
  @test !isa(cat(rand(2), dims=1), TrackedArray)

  @test gradtest((a,b)->cat(a, b, dims = (2,3,5)), rand(2,3), rand(2,4,2,1))

  @testset "promotiontest" begin
    @testset for fcat in [hcat, vcat, (x...) -> cat(x..., dims = 3), (x...) -> cat(x..., dims = (1,2))]
      promotiontest(fcat, rand(2), rand(2), rand(2))
      promotiontest(fcat, rand(2)', rand(2)', rand(2)')
      promotiontest(fcat, rand(2,2), rand(2,2), rand(2,2))
      promotiontest(fcat, rand(2,2,2), rand(2,2,2), rand(2,2,2))
    end

    promotiontest(vcat, rand(1,2), rand(2)', rand(2,2))
    promotiontest(hcat, rand(2,1), rand(2), rand(2,2))
    promotiontest(vcat, rand(3,4,5), rand(1,4,5), rand(2,4,5))
    promotiontest(hcat, rand(4,3,5), rand(4,1,5), rand(4,2,5))
    promotiontest((x...) -> cat(x..., dims = 3), rand(4,5,3), rand(4,5,1), rand(4,5,2))
  end

  @testset "scalars" begin
    @test vcat(param([1, 2, 3]), 1) isa TrackedArray
    @test vcat(1, param([1, 2, 3])) isa TrackedArray
    @test hcat(1, param([1 2 3;])) isa TrackedArray
    @test vcat(param(1), 2) isa TrackedArray
  end

end

@testset "getindex (Nabla.jl - #139)" begin
  z = [2, 3, 3]
  @test gradtest(x->x[z], randn(MersenneTwister(123456), 3))
end

@test gradtest(x -> permutedims(x, [3,1,2]), rand(4,5,6))
@test gradtest(x -> PermutedDimsArray(x, [3,1,2]), rand(4,5,6))

@test gradtest(reverse, rand(5))
@test gradtest(x -> reverse(x, dims=2), rand(4,5,6))
@test gradtest(x -> reverse(x, 2, 4), rand(5))

@test gradtest(x -> repeat(x; inner=2), rand(5))
@test gradtest(x -> repeat(x; inner=2, outer=3), rand(5))
@test gradtest(x -> repeat(x; inner=(2,2,1), outer=(1,1,3)), rand(5,4,3))

@test gradtest(kron, rand(5), rand(3))
@test gradtest(kron, rand(5), rand(3), rand(8))
@test gradtest(kron, rand(5,1), rand(3,1))
@test gradtest(kron, rand(5,1), rand(3,1), rand(8,1))
@test gradtest(kron, rand(5,2), rand(3,2), rand(8,2))

@test gradtest(x -> diagm(0 => x), rand(3))

@test gradtest(W -> inv(log.(W * W)), (5,5))
@test gradtest((A, B) -> A / B , (1,5), (5,5))
@test gradtest((A, B) -> log.(A * A) / exp.(B * B), (5,5), (5,5))
@test gradtest((A, B) -> A \ B, (5, 5), (5,))
@test let A=rand(5, 5)
    gradtest(B -> A \ B, (5,))
end
@test let B=rand(5,)
    gradtest(A -> A \ B, (5, 5))
end
@test gradtest((A, B) -> log.(A * A) \ exp.(B * B), (5,5), (5,5))
@test let A=rand(5, 5)
    gradtest(B -> log.(A * A) \ exp.(B * B), (5, 5))
end
@test let B=rand(5, 5)
    gradtest(A -> log.(A * A) \ exp.(B * B), (5, 5))
end

@testset "mean" begin
  @test gradtest(mean, rand(2, 3))

  @test gradtest(x -> mean(x, dims=1), rand(2, 3))
  @test gradtest(x -> mean(x, dims=2), rand(2, 3))
  @test gradtest(x -> mean(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> mean(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "maximum" begin
  @test gradtest(maximum, rand(2, 3))

  @test gradtest(x -> maximum(x, dims=1), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=2), rand(2, 3))
  @test gradtest(x -> maximum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> maximum(x, dims=[1, 2]), rand(2, 3, 4))
end

@testset "minimum" begin
  @test gradtest(minimum, rand(2, 3))

  @test gradtest(x -> minimum(x, dims=1), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=2), rand(2, 3))
  @test gradtest(x -> minimum(x, dims=3), rand(2, 3, 4))

  @test gradtest(x -> minimum(x, dims=[1, 2]), rand(2, 3, 4))
end

@test gradtest(x -> std(x), rand(5,5))
@test gradtest(x -> std(x, dims = 1), rand(5,5))
@test gradtest(x -> std(x, dims = 1, corrected = false), rand(5,5))

@test gradtest((x, y) -> x .* y, rand(5), rand(5))
@test gradtest(dot, rand(5), rand(5))

@test gradtest(norm, rand(5))
@test gradtest(norm, zeros(5))
@test norm(TrackedArray(rand(Float32, 5))) isa TrackedReal{Float32}

@test gradtest(rand(5)) do x
  y = x.^2
  2y + x
end

@testset "transpose" begin
  w = Tracker.TrackedArray(rand(5,5))
  x = Tracker.TrackedArray(rand(5,5))
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray

  x = Tracker.TrackedArray(rand(5))
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  w = Tracker.TrackedArray(rand(5))
  x = Tracker.TrackedArray(rand(5))
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  w = Tracker.TrackedArray(rand(5))
  x = Tracker.TrackedArray(rand(5,5))
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray

  w = Tracker.TrackedArray(rand(5,5))
  x = rand(5,5)
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray
  x = rand(5)
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  w = Tracker.TrackedArray(rand(5))
  x = rand(5)
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  w = Tracker.TrackedArray(rand(5))
  x = rand(5,5)
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray

  w = rand(5,5)
  x = Tracker.TrackedArray(rand(5,5))
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray
  x = Tracker.TrackedArray(rand(5))
  @test w*x   isa TrackedArray
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  w = rand(5)
  x = Tracker.TrackedArray(rand(5))
  @test w*x'  isa TrackedArray
  @test w*transpose(x)  isa TrackedArray
  w = rand(5)
  x = Tracker.TrackedArray(rand(5,5))
  @test w'*x  isa TrackedArray
  @test transpose(w)*x  isa TrackedArray
  @test w'*x' isa TrackedArray
  @test transpose(w)*transpose(x) isa TrackedArray
end

@testset "conv" begin
  for spatial_rank in (1, 2, 3)
    x = rand(repeat([10], spatial_rank)..., 3, 2)
    w = rand(repeat([3], spatial_rank)..., 3, 3)
    cdims = DenseConvDims(x, w)
    @test gradtest((x, w) -> conv(x, w, cdims), x, w)
    y = conv(x, w, cdims)
    @test gradtest((y, w) -> ∇conv_data(y, w, cdims), y, w)
    dcdims = DepthwiseConvDims(x, w)
    @test gradtest((x, w) -> depthwiseconv(x, w, dcdims), x, w)
    end
end

@testset "pooling" begin
  for spatial_rank in (1, 2)
    x = rand(repeat([10], spatial_rank)..., 3, 2)
    pdims = PoolDims(x, 2)
    @test gradtest(x -> maxpool(x, pdims), x)
    @test gradtest(x -> meanpool(x, pdims), x)
  end
end


@test gradtest(x -> Float64.(x), 5)

@testset "equality & order" begin
    # TrackedReal
    @test param(2)^2 == param(4)
    @test param(2)^2 == 4
    @test 4 == param(2)^2

    @test param(2)^2 ≈ param(4)
    @test param(2)^2 ≈ 4
    @test 4 ≈ param(2)^2

    @test (param([1,2,3]) .< 2) == [true, false, false]
    @test (param([1,2,3]) .<= 2) == [true, true, false]
    @test (2 .> param([1,2,3])) == [true, false, false]
    @test (2 .>= param([1,2,3])) == [true, true, false]

    # TrackedArray
    @test param([1,2,3]).^2 == param([1,4,9])
    @test [1,2,3].^2 == param([1,4,9])
    @test param([1,2,3]).^2 == [1,4,9]

    @test param([1,2,3]).^2 ≈ param([1,4,9])
    @test [1,2,3].^2 ≈ param([1,4,9])
    @test param([1,2,3]).^2 ≈ [1,4,9]
end

@testset "reshape" begin
  x = reshape(param(rand(2,2,2)), 4, 2)
  @test x isa TrackedArray
  @test size(x) == (4,2)
  x = reshape(param([1]), (1,:))
  @test x isa TrackedArray
  @test size(x) == (1,1)
  x = reshape(param(rand(2)), (2,:))
  @test x isa TrackedArray
  @test size(x) == (2,1)
  x = reshape(param(rand(2,2)), (1,:,2))
  @test x isa TrackedArray
  @test size(x) == (1,2,2)
end

@testset "Intermediates" begin
  x = param([1])
  l = sum((x .+ x).^2)
  Tracker.back!(l, once = false)
  @test x.grad == [8]
  x.grad .= 0
  Tracker.back!(l, once = false)
  @test x.grad == [8]
end

@testset "Fallbacks" begin
  xs = param([1 2; 3 4])
  @test similar(xs) isa Matrix{Float64}
end

@test @sprintf("%.2f", sum(param([1,2,3]))) == "6.00"

# This no longer infers cleanly :(
#@inferred NNlib.conv(param(rand(10,10,3,2)), randn(Float64,2,2,3,4), DenseConvDims((10,10,3,2),(2,2,3,4)))

b = param(rand())
Tracker.back!(b)
@test Tracker.grad(b) == 1

@testset "collect" begin
  x, y = param(2), param(3)
  xy = Tracker.collect([x, y])
  @test xy isa TrackedArray{Float64}
  z = xy[1]*xy[2]
  back!(z)
  @test grad.((x,y)) == (3, 2)

  @test gradient(2, 3) do x, y
    xy = Tracker.collect([x, y])
    xy[1]*xy[2]
  end == (3, 2)

  gradient(x -> sum(Tracker.collect(x)), [1, 2, 3])[1] == [1, 1, 1]
end

# Gradient Hooks
@testset "Hooks" begin
  x = param(2)
  y = Tracker.hook(-, x)
  back!(y)
  @test grad(x) == -1
end

@testset "Checkpointing" begin
  count = 0
  function mul(a, b)
    count += 1
    a * b
  end
  @test gradient(x -> mul(5, x), 3)[1] == 5
  @test count == 1
  @test gradient(x -> checkpoint(mul, 5, x), 3)[1] == 5
  @test count == 3
end

@testset "Updates" begin
  xs = param([1, 2, 3])
  Tracker.update!(xs, param([4, 5, 6]))
  @test xs == [5, 7, 9]
  x = param(3)
  Tracker.update!(x, param(4))
  @test x == 7
end

@testset "Params" begin
  W = param(randn(5, 10))
  x = rand(10)
  dW = gradient(W -> sum(W*x), W)[1]
  gs = gradient(() -> sum(W*x), Tracker.Params([W]))
  @test gs[W] == dW
end

@testset "Forward" begin
  @test @inferred(Tracker.forward_jacobian(x -> [sum(x)], rand(5,5), Val(12)))[2] ==
    reshape(ones(25), :, 1)
  @test gradient([2, 3]) do x
    forwarddiff(x) do x
      x[1]*x[2]
    end
  end == ([3, 2],)
end

@testset "Custom Sensitivities" begin
  y, back = Tracker.forward(x -> [3x^2, 2x], 5)
  @test back([1, 1]) == (32,)
end

@testset "PDMats" begin
    B = rand(5, 5)
    S = PDMat(I + B * B')
    @test gradtest(A -> S / A, (5, 5))

    S = PDiagMat(rand(5))
    @test gradtest(A -> S / A, (5, 5))

    S = ScalMat(5, rand())
    @test gradtest(A -> S / A, (5, 5))
end


@testset "broadcast" begin
    f(x) = @. x < zero(x)
    y = @inferred(f(TrackedArray(rand(5, 3))))
    @test size(y) == (5, 3)
    @test !any(y)

    g(x) = @. (x < zero(x)) * x
    y = @inferred(g(TrackedArray(rand(5, 3))))
    @test size(y) == (5, 3)
end

end #testset
