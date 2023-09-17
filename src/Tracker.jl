module Tracker

using MacroTools
using MacroTools: @q, @forward

using DiffRules
using ForwardDiff
import LogExpFunctions
import NaNMath
import SpecialFunctions

import Printf

import Base: ==

export TrackedArray, TrackedVector, TrackedMatrix, Params, gradient,
  jacobian, hessian, param, back!, withgradient

tracker(x) = nothing

istracked(x) = tracker(x) ≠ nothing
isleaf(x) = !istracked(x) || isleaf(tracker(x))
grad(x) = grad(tracker(x))
grad(::Nothing) = nothing
data(x) = x

"""
  Call{F,As<:Tuple}

Structure to keep the function `func`::F and it's arguments `args`.
"""
struct Call{F,As<:Tuple}
  func::F
  args::As
end

Call(f::F, args::T) where {F,T} = Call{F,T}(f, args)
Call() = Call(nothing, ())

# When deserialising, the object_id changes
a::Call == b::Call = a.func == b.func && a.args == b.args

@inline (c::Call)() = c.func(data.(c.args)...)

"""
  Tracked{T}
  
Structure used to keep the operations applied over variables. 
Represents a node in the graph. To navigate in the graph, use the `f::Call` field. 

# Parameters
  - `ref`: variable used during the graph traversals, how many times we reached a node
  - `f::Call`: the Call object containing the recorded function and arguments; kindly note the pullback function is stored instead
              of the original function; e.g. we store the pullback of + and not the + function itself
  - `isleaf::Bool`: refers to the node in the built graphs; true if the node (tracked object) is leaf
  - `grad::T`: use to store the value of the back-propagated gradient. 
               To further propagate this gradient, let's call it `∇`, the algorithm applies the Jacobian `∇2 = f.func(∇) = J(f_original)*∇` (the pullback). 
               This new gradient is passed to the "children" of `f` stored in `f.args`.               
               Note the gradient is not always stored. 
               For example if the graph is just a straigh-line, no branches, then we simply back-propagate the gradients 
               from the output to the input params. Only the leafs in the graph (our input params) will store gradients in this case.
               See the `function back(x::Tracked, Δ, once)` for more details.
"""
mutable struct Tracked{T}
  ref::UInt32  
  f::Call
  isleaf::Bool 
  grad::T
  Tracked{T}(f::Call) where T = new(0, f, false)
  Tracked{T}(f::Call, grad::T) where T = new(0, f, false, grad)
  Tracked{T}(f::Call{Nothing}, grad::T) where T = new(0, f, true, grad)
end

istracked(x::Tracked) = true
isleaf(x::Tracked) = x.f == Call()
grad(x::Tracked) = x.grad

track(f::Call, x) = Tracked{typeof(x)}(f)

function _forward end

function track(f::F, xs...; kw...) where F
  y, back = _forward(f, xs...; kw...)
  track(Call(back, tracker.(xs)), y)
end

macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end

include("idset.jl")
include("params.jl")
include("lib/real.jl")
include("lib/array.jl")
include("back.jl")
include("numeric.jl")
include("forward.jl")

if !isdefined(Base, :get_extension)
  using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
  @require PDMats="90014a1f-27ba-587c-ab20-58faa44d9150" include("../ext/TrackerPDMatsExt.jl")
end
end


"""
    hook(f, x) -> x′

Hook into gradient backpropagation. `x` is unmodified, but when backpropagating
`f` will be applied to the incoming gradient. For example, `hook(-, x)` will reverse
the sign of the gradient applied to `x`.
"""
hook(f, x) = istracked(x) ? track(hook, f, x) : x
@grad hook(f, x) = data(x), Δ -> (nothing, f(Δ))

"""
    checkpoint(f, args...)

Behaves like `f(args...)`, but avoids storing the intermediate values needed for
calculating gradients. Instead, `f(args...)` will be called again during the
backward pass. This can be used to save memory in larger models.
"""
checkpoint(f, args...) = track(checkpoint, f, args...)

@grad function checkpoint(f, args...)
  data(f(args...)), function (Δ)
    y, back = forward(f, args...)
    (nothing, back(Δ)...)
  end
end

nobacksies(f, x) = track(nobacksies, f, x)
nobacksies(f, xs::Tuple) = map(x -> nobacksies(f, x), xs)
@grad nobacksies(f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
@grad nobacksies(f::String, x) = data(x), Δ -> error(f)

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

@grad identity(x) = data(x), Δ -> (Δ,)
param(x::TrackedReal) = track(identity, x)
param(x::TrackedArray) = track(identity, x)

import Adapt: adapt, adapt_structure

adapt_structure(T, xs::TrackedArray) = param(adapt(T, data(xs)))

end
