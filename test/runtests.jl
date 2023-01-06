using Tracker, Test, Random, Statistics

Random.seed!(0)

@testset "Tracker" begin

include("tracker.jl")

using Tracker: jacobian

@testset "Jacobian" begin
  A = param(randn(2,2))
  x = randn(2)
  m(x) = A*x
  y = m(x)
  J = jacobian(m,x)
  @test J ≈ A.data
end

using Optimisers, Functors
struct TwoThirds a; b; c; end  # evil test from Optimisers.jl
@eval Functors.@functor TwoThirds (a, c)
Optimisers.trainable(x::TwoThirds) = (a = x.a,)

@testset "withgradient" begin
  nt = (vec = [1.0, 2.0], mat = [4.0;;], fun = sin);
  @test withgradient((x, p) -> sum(abs2, x.vec) ^ p, nt, 2) == (val = 25.0, grad = ((vec = [20.0, 40.0], mat = [0.0;;], fun = nothing), nothing))

  @test withgradient(x -> sum(x.v), (v = [1, 2], w = [3.0])) == (val = 3, grad = nothing)

  m = TwoThirds([1.0], [2.0], [3.0])  # only the first should be tracked, but all should survive
  g = withgradient(m -> only(m.a::AbstractVector + m.b::Vector + m.c::Vector), m)
  @test g == (val = 6.0, grad = ((a = [1.0], b = nothing, c = nothing),))
end

using NNlib
@testset "NNlib.within_gradient" begin
  f_good(x) = NNlib.within_gradient(x) ? 10x : x
  @test gradient(f_good, 1.0)[1] == 10
  @test gradient(x -> sum(f_good(x)), [1.0])[1] == [10]
end

end  # overall @testset
