import LogExpFunctions

@testset "LogExpFunctions" begin
    x = 0.4
    @test gradient(LogExpFunctions.logit, x)[1] ≈ inv(x * (1 - x))

    x = 0.2
    @test gradient(LogExpFunctions.log1pmx, x)[1] ≈ -x / (1 + x)

    x = -0.3
    @test gradient(LogExpFunctions.log2mexp, x)[1] ≈ -exp(x) / (2 - exp(x))

    x, y = 0.2, -0.4
    ∂x, ∂y = gradient(LogExpFunctions.logaddexp, x, y)
    denominator = exp(x) + exp(y)
    @test ∂x ≈ exp(x) / denominator
    @test ∂y ≈ exp(y) / denominator
end
