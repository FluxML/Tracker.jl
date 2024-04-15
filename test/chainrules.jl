# Adapted from https://github.com/JuliaDiff/Tracker.jl/blob/master/test/ChainRulesTests.jl
module ChainRulesTest  # Run in isolatex environment

using LinearAlgebra
using ChainRulesCore
using Tracker
using Test

struct MyStruct end
f(::MyStruct, x) = sum(4x .+ 1)
f(x, y::MyStruct) = sum(4x .+ 1)
f(x) = sum(4x .+ 1)

function ChainRulesCore.rrule(::typeof(f), x)
  r = f(x)
  function back(d)
    #=
    The proper derivative of `f` is 4, but in order to
    check if `ChainRulesCore.rrule` had taken over the compuation,
    we define a rrule that returns 3 as `f`'s derivative.

    After importing this rrule into Tracker, if we get 3
    rather than 4 when we compute the derivative of `f`, it means
    the importing mechanism works.
    =#
    return NoTangent(), fill(3 * d, size(x))
  end
  return r, back
end
function ChainRulesCore.rrule(::typeof(f), ::MyStruct, x)
  r = f(MyStruct(), x)
  function back(d)
    return NoTangent(), NoTangent(), fill(3 * d, size(x))
  end
  return r, back
end
function ChainRulesCore.rrule(::typeof(f), x, ::MyStruct)
  r = f(x, MyStruct())
  function back(d)
    return NoTangent(), fill(3 * d, size(x)), NoTangent()
  end
  return r, back
end

Tracker.@grad_from_chainrules f(x::Tracker.TrackedArray)
# test arg type hygiene
Tracker.@grad_from_chainrules f(::MyStruct, x::Tracker.TrackedArray)
Tracker.@grad_from_chainrules f(x::Tracker.TrackedArray, y::MyStruct)

g(x, y) = sum(4x .+ 4y)

function ChainRulesCore.rrule(::typeof(g), x, y)
  r = g(x, y)
  function back(d)
    # same as above, use 3 and 5 as the derivatives
    return NoTangent(), fill(3 * d, size(x)), fill(5 * d, size(x))
  end
  return r, back
end

Tracker.@grad_from_chainrules g(x::Tracker.TrackedArray, y)
Tracker.@grad_from_chainrules g(x, y::Tracker.TrackedArray)
Tracker.@grad_from_chainrules g(x::Tracker.TrackedArray, y::Tracker.TrackedArray)

@testset "rrule in ChainRules and Tracker" begin
  ## ChainRules
  # function f
  input = rand(3, 3)
  output, back = ChainRulesCore.rrule(f, input)
  _, d = back(1)
  @test output == f(input)
  @test d == fill(3, size(input))
  # function g
  inputs = rand(3, 3), rand(3, 3)
  output, back = ChainRulesCore.rrule(g, inputs...)
  _, d1, d2 = back(1)
  @test output == g(inputs...)
  @test d1 == fill(3, size(inputs[1]))
  @test d2 == fill(5, size(inputs[2]))
end

@testset "custom struct input" begin
  input = rand(3, 3)
  output, back = ChainRulesCore.rrule(f, MyStruct(), input)
  _, _, d = back(1)
  @test output == f(MyStruct(), input)
  @test d == fill(3, size(input))

  output, back = ChainRulesCore.rrule(f, input, MyStruct())
  _, d, _ = back(1)
  @test output == f(input, MyStruct())
  @test d == fill(3, size(input))
end

### Functions with varargs and kwargs
# Varargs
f_vararg(x, args...) = sum(4x .+ sum(args))

function ChainRulesCore.rrule(::typeof(f_vararg), x, args...)
  r = f_vararg(x, args...)
  function back(d)
    return (NoTangent(), fill(3 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...)
  end
  return r, back
end

Tracker.@grad_from_chainrules f_vararg(x::Tracker.TrackedArray, args...)

@testset "Function with Varargs" begin
  grads = Tracker.gradient(x -> f_vararg(x, 1, 2, 3) + 2, rand(3, 3))

  @test grads[1] == fill(3, (3, 3))
end

# Vargs and kwargs
f_kw(x, args...; k=1, kwargs...) = sum(4x .+ sum(args) .+ (k + kwargs[:j]))

function ChainRulesCore.rrule(::typeof(f_kw), x, args...; k=1, kwargs...)
  r = f_kw(x, args...; k=k, kwargs...)
  function back(d)
    return (NoTangent(), fill(3 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...)
  end
  return r, back
end

Tracker.@grad_from_chainrules f_kw(x::Tracker.TrackedArray, args...; k=1, kwargs...)

@testset "Function with Varargs and kwargs" begin
  inputs = rand(3, 3)
  results = Tracker.gradient(x -> f_kw(x, 1, 2, 3; k=2, j=3) + 2, inputs)

  @test results[1] == fill(3, size(inputs))
end

### Mix @grad and @grad_from_chainrules

h(x) = 10x
h(x::Tracker.TrackedArray) = Tracker.track(h, x)
Tracker.@grad function h(x)
  xv = Tracker.data(x)
  return h(xv), Δ -> (Δ * 7,) # use 7 asits derivatives
end

@testset "Tracker and ChainRules Mixed" begin
  t(x) = g(x, h(x))
  inputs = rand(3, 3)
  results = Tracker.gradient(t, inputs)
  @test results[1] == fill(38, size(inputs)) # 38 = 3 + 5 * 7
end

### Isolated Scope
module IsolatedModuleForTestingScoping
using ChainRulesCore
using Tracker: Tracker, @grad_from_chainrules

f(x) = sum(4x .+ 1)

function ChainRulesCore.rrule(::typeof(f), x)
  r = f(x)
  function back(d)
    # return a distinguishable but improper grad
    return NoTangent(), fill(3 * d, size(x))
  end
  return r, back
end

@grad_from_chainrules f(x::Tracker.TrackedArray)

module SubModule
using Test
using Tracker: Tracker
using ..IsolatedModuleForTestingScoping: f
@testset "rrule in Isolated Scope" begin
  inputs = rand(3, 3)
  results = Tracker.gradient(x -> f(x) + 2, inputs)

  @test results[1] == fill(3, size(inputs))
end

end # end of SubModule
end # end of IsolatedModuleForTestingScoping

end