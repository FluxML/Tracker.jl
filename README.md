# Tracker.jl

[![Build Status](https://github.com/FluxML/Tracker.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/FluxML/Tracker.jl/actions/workflows/CI.yml?query=branch%3Amaster)
[![Coverage](https://codecov.io/gh/FluxML/Tracker.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/FluxML/Tracker.jl)
<!---[![Coverage](https://coveralls.io/repos/github/FluxML/Tracker.jl/badge.svg?branch=master)](https://coveralls.io/github/FluxML/Tracker.jl?branch=master) --->
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)


This was the original automatic differentiation engine for [Flux.jl](https://github.com/FluxML/Flux.jl), before being replaced by [Zygote.jl](https://github.com/FluxML/Zygote.jl) in 2019. Both were written by Mike Innes.

This package is solid and still in active use, but is no longer heavily maintained. PRs and issues may go unanswered.

### Introduction

Like [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) and [AutoGrad.jl](https://github.com/denizyuret/AutoGrad.jl), Tracker traces through a program by wrapping arrays in a special `TrackedArray` type. The final answer contains a "tape" of the operations performed, which is reversed by `back!`:

```julia
x = param([1,2,3])  # Tracked 3-element Vector{Float64}

f(x) = sum(abs2, x) + prod(x[2:end])

y = f(x)  # TrackedReal

back!(y)  # run back-propagation

Tracker.grad(x)  # extract gradient from TrackedArray
```

This is a much lower-tech approach than that of [Zygote](https://github.com/FluxML/Zygote.jl), [Yota](https://github.com/dfdx/Yota.jl) and [Diffractor](https://github.com/JuliaDiff/Diffractor.jl). At best, those can produce fast, compiled Julia code for the reverse pass, instead of an interpreted tape. At worst, they can have extremely long compile-times and can be difficult to debug.

### Interface

Instead of calling `back!` yourself, you can pass the function and the input to `gradient`:

```julia
gradient(f, [1,2,3])  # returns ([2.0, 7.0, 8.0],)

withgradient(f, [1,2,3])  # returns (val = 20, grad = ([2.0, 7.0, 8.0],))
```

The original interface to [Flux.jl](https://github.com/FluxML/Flux.jl) involved a dictionary of arrays called `Params`, much like Zygote's "implicit" parameter interface. This appears not to be documented.

A more modern way to use Flux relies on `withgradient`'s ability to take gradients with respect to complex nested structures. This is what [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) is designed to accept:

```julia
julia> using Flux, Tracker

julia> model = Chain(Dense(2 => 1, tanh), Dense(1 => 1, bias=false));

julia> withgradient(model, rand(Float32, 2)) do m, x
         sum(abs2, m(x))
       end
(val = 0.035716165f0, 
 grad = ((layers = ((weight = Float32[-0.4241869 -0.16741231], bias = Float32[-0.5529184], σ = nothing), 
                    (weight = Float32[-0.04804218;;], bias = nothing, σ = nothing)),), 
         Float32[0.12706584, -0.08858479]))
```

### Rules

Tracker.jl contains rules for many common operations. It relies on [DiffRules.jl](https://github.com/JuliaDiff/DiffRules.jl) for many definitions, and does not connect to the newer [ChainRules.jl](https://github.com/JuliaDiff/ChainRules.jl) at all.

To define more rules, use `track` and `@grad`. See the source for more examples:

```julia
f(x::TrackedArray) = track(f, x)    # entry point, via dispatch

@grad function f(x)
  y = f(data(x))                    # forward pass, withtout tracking
  back(dy) = (dy * ∂f∂x(data(x)),)  # pullback function, returns a tuple
  return y, back
end
```

