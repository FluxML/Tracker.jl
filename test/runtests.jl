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

end
