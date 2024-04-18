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

rrule_f_singleargs = Ref(0)
rrule_f_mystruct_x = Ref(0)
rrule_f_x_mystruct = Ref(0)

function ChainRulesCore.rrule(::typeof(f), x)
  rrule_f_singleargs[] += 1
  r = f(x)
  back(d) = NoTangent(), fill(4 * d, size(x))
  return r, back
end
function ChainRulesCore.rrule(::typeof(f), ::MyStruct, x)
  rrule_f_mystruct_x[] += 1
  r = f(MyStruct(), x)
  back(d) = NoTangent(), NoTangent(), fill(4 * d, size(x))
  return r, back
end
function ChainRulesCore.rrule(::typeof(f), x, ::MyStruct)
  rrule_f_x_mystruct[] += 1
  r = f(x, MyStruct())
  back(d) = NoTangent(), fill(4 * d, size(x)), NoTangent()
  return r, back
end

Tracker.@grad_from_chainrules f(x::Tracker.TrackedArray)
# test arg type hygiene
Tracker.@grad_from_chainrules f(::MyStruct, x::Tracker.TrackedArray)
Tracker.@grad_from_chainrules f(x::Tracker.TrackedArray, y::MyStruct)

g(x, y) = sum(4x .+ 4y)

rrule_g_x_y = Ref(0)

function ChainRulesCore.rrule(::typeof(g), x, y)
  rrule_g_x_y[] += 1
  r = g(x, y)
  back(d) = NoTangent(), fill(4 * d, size(x)), fill(4 * d, size(x))
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
  @test d == fill(4, size(input))
  @test rrule_f_singleargs[] == 1
  # function g
  inputs = rand(3, 3), rand(3, 3)
  output, back = ChainRulesCore.rrule(g, inputs...)
  _, d1, d2 = back(1)
  @test output == g(inputs...)
  @test d1 == fill(4, size(inputs[1]))
  @test d2 == fill(4, size(inputs[2]))
  @test rrule_g_x_y[] == 1
end

@testset "custom struct input" begin
  input = rand(3, 3)
  output, back = ChainRulesCore.rrule(f, MyStruct(), input)
  _, _, d = back(1)
  @test output == f(MyStruct(), input)
  @test d == fill(4, size(input))
  @test rrule_f_mystruct_x[] == 1

  output, back = ChainRulesCore.rrule(f, input, MyStruct())
  _, d, _ = back(1)
  @test output == f(input, MyStruct())
  @test d == fill(4, size(input))
  @test rrule_f_x_mystruct[] == 1
end

### Functions with varargs and kwargs
# Varargs
f_vararg(x, args...) = sum(4x .+ sum(args))

rrule_f_vararg = Ref(0)

function ChainRulesCore.rrule(::typeof(f_vararg), x, args...)
  rrule_f_vararg[] += 1
  r = f_vararg(x, args...)
  back(d) = (NoTangent(), fill(4 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...)
  return r, back
end

Tracker.@grad_from_chainrules f_vararg(x::Tracker.TrackedArray, args...)

@testset "Function with Varargs" begin
  grads = Tracker.gradient(x -> f_vararg(x, 1, 2, 3) + 2, rand(3, 3))

  @test grads[1] == fill(4, (3, 3))
  @test rrule_f_vararg[] == 1
end

# Vargs and kwargs
f_kw(x, args...; k=1, kwargs...) = sum(4x .+ sum(args) .+ (k + kwargs[:j]))

rrule_f_kw = Ref(0)

function ChainRulesCore.rrule(::typeof(f_kw), x, args...; k=1, kwargs...)
  rrule_f_kw[] += 1
  r = f_kw(x, args...; k=k, kwargs...)
  back(d) = (NoTangent(), fill(4 * d, size(x)), ntuple(_ -> NoTangent(), length(args))...)
  return r, back
end

Tracker.@grad_from_chainrules f_kw(x::Tracker.TrackedArray, args...; k=1, kwargs...)

@testset "Function with Varargs and kwargs" begin
  inputs = rand(3, 3)
  results = Tracker.gradient(x -> f_kw(x, 1, 2, 3; k=2, j=3) + 2, inputs)

  @test results[1] == fill(4, size(inputs))
  @test rrule_f_kw[] == 1
end

### Mix @grad and @grad_from_chainrules

h(x) = 10x
h(x::Tracker.TrackedArray) = Tracker.track(h, x)

grad_hcalls = Ref(0)

Tracker.@grad function h(x)
  grad_hcalls[] += 1
  xv = Tracker.data(x)
  return h(xv), Δ -> (Δ * 10,) # use 7 asits derivatives
end

@testset "Tracker and ChainRules Mixed" begin
  t(x) = g(x, h(x))
  inputs = rand(3, 3)
  results = Tracker.gradient(t, inputs)
  @test results[1] == fill(44, size(inputs)) # 44 = 4 + 4 * 10
  @test rrule_g_x_y[] == 2
  @test grad_hcalls[] == 1
end

### Isolated Scope
module IsolatedModuleForTestingScoping

using ChainRulesCore, Test
using Tracker: Tracker, @grad_from_chainrules

f(x) = sum(4x .+ 1)

rrule_f_singleargs = Ref(0)

function ChainRulesCore.rrule(::typeof(f), x)
  rrule_f_singleargs[] += 1
  r = f(x)
  back(d) = NoTangent(), fill(4 * d, size(x))
  return r, back
end

@grad_from_chainrules f(x::Tracker.TrackedArray)

module SubModule
using Test
using Tracker: Tracker
using ..IsolatedModuleForTestingScoping: f, rrule_f_singleargs

@testset "rrule in Isolated Scope" begin
  inputs = rand(3, 3)
  results = Tracker.gradient(x -> f(x) + 2, inputs)

  @test results[1] == fill(4, size(inputs))
  @test rrule_f_singleargs[] == 1
end

end # end of SubModule

end # end of IsolatedModuleForTestingScoping

end