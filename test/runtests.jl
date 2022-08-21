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

@testset "withgradient" begin
  nt = (vec = [1.0, 2.0], mat = [4.0;;], fun = sin);
  @test withgradient((x, p) -> sum(abs2, x.vec) ^ p, nt, 2) == (val = 25.0, grad = ((vec = [20.0, 40.0], mat = [0.0;;], fun = nothing), nothing))

  @test withgradient(x -> sum(x.v), (v = [1, 2], w = [3.0])) == (val = 3, grad = nothing)
end

end  # overall @testset
