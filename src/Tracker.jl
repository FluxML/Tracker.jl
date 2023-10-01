module Tracker

using MacroTools
using MacroTools: @q, @forward

using ChainRules
using ChainRules: rrule, RuleConfig, HasReverseMode
using ForwardDiff
import LogExpFunctions
import NaNMath
import SpecialFunctions

import Printf

import Base: ==
import Base: broadcasted

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

track_ctor(f::Call, x) = Tracked{typeof(x)}(f)

function _forward end

# TODO: this function is used to define gradients for a couple of functions, especially in arrays, which are not used,
# but we might want to define rrules for them, so we keep this code for a while
macro grad(ex)
  @capture(shortdef(ex), (name_(args__) = body_) |
                         (name_(args__) where {T__} = body_)) || error("Need a function definition")
  T == nothing && (T = [])
  isexpr(name, :(::)) || (name = :(::typeof($name)))
  insert!(args, 1+isexpr(args[1], :parameters) , name)
  @q(Tracker._forward($(args...)) where $(T...) = $body) |> esc
end

if !isdefined(Base, :get_extension)
  using Requires
end

@static if !isdefined(Base, :get_extension)
function __init__()
  @require PDMats="90014a1f-27ba-587c-ab20-58faa44d9150" include("../ext/TrackerPDMatsExt.jl")
end
end

include("idset.jl")
include("params.jl")
include("lib/real.jl")
include("lib/array.jl")
include("back.jl")
include("numeric.jl")
include("forward.jl")

TrackedTypes = Union{TrackedReal, TrackedArray, TrackedTuple}

# we define this in order to access rrule for broadcasted
struct TrackerRuleConfig <: RuleConfig{HasReverseMode} end
const tracker_rule_cfg = TrackerRuleConfig()
const dummy_broadcast_style = Base.BroadcastStyle(Float64) # requested by rrule for broadcasted, which is only to please Zygote

# dedicated track method for broadcasted
function track(bf::typeof(Base.broadcasted), f::F, xs...; kw...) where F
  @info "Chainrules for $bf($f, $xs)"
  y, _back = rrule(tracker_rule_cfg, bf, dummy_broadcast_style, f, data.(xs)...; kw...)
  back = Δ->_back(Δ)[4:end] # TODO: what happens if f is a struct?
  track_ctor(Call(back, tracker.(xs)), y)
end

# Arithmetic operations +, -, *, ^ have a dedicated specializations in ChainRules; are these faster? we use them here
for f in (:+, :-, :*, :/)
  @eval begin
    function track(bf::typeof(Base.broadcasted), ::typeof($f), xs...; kw...)
      @info "Chainrules for $bf($($f), $xs), specialized"
      _y, _back = rrule(bf, $f, data.(xs)...; kw...)
      y = Base.materialize(_y)
      back = Δ->_back(Δ)[3:end]
      track_ctor(Call(back, tracker.(xs)), y)
    end
  end
end

# ^2 also has a dedicated specialization in ChainRules
function track(bf::typeof(Base.broadcasted), lp::typeof(Base.literal_pow), ::typeof(^), x::TrackedTypes, ::Val{2})
  @info "Chainrules for $bf($lp, ^, $x, 2), specialized"
  _y, _back = rrule(bf, lp, ^, data(x), Val(2))
  y = Base.materialize(_y)
  back = Δ->_back(Δ)[4:4] # 4:4 because the output shall be a tuple, not a scalar
  track_ctor(Call(back, (tracker(x), )), y)
end

# TODO: we can better define a method to select the range of interested values, e.g. without NoTangent()
# option1: specialize it for various methods
# option2: simply scan the result and select values without NoTangent(), shall be consecutive

function track(::typeof(Base.getindex), xs...; kw...)
  @assert length(xs) == 2 # the array and the index
  @info "Chainrules for Base.getindex"
  # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
  y, _back = rrule(Base.getindex, data.(xs)...; kw...)
  back = Δ->_back(Δ)[2:2]
  if typeof(xs[1]) <: TrackedTuple # the rrule getindex from Tuples returns a Tangent{..}(result), compared to arrays where it returns directly the result
    back = Δ->(ChainRules.ChainRulesCore.backing(_back(Δ)[2]),)
  end
  track_ctor(Call(back, tracker.(xs[1:1])), y)   
  # TODO: only tracker.(xs[1:1]), tracker(index) is nothing; hm... use the operations on NoTangent() and avoid all this special treatment?
end

function track(f::F, xs...; kw...) where F
  @info "Chainrules for $f"
  # untracked primal y; also untracked pullback back as we rrule over the data.(xs)
  y, _back = rrule(f, data.(xs)...; kw...)
  # the rrule pullback returns (NoTangent(), out_grads) for functions
  # TODO: what happens with structs as functions?
  back = Δ->_back(Δ)[2:end]
  track_ctor(Call(back, tracker.(xs)), y)
end


"Function to print the graph of an Call"
function print_graph_(io::IO, f::Call, indent, indent_all)
  println(io, indent_all*"Call=")
  indent_all *= indent

  println(io, indent_all, "f")
  if !isnothing(f.func) && !isnothing(f.func)
    println(io, indent*indent_all, f.func)
    fnms = fieldnames(typeof(f.func))
    for fnm in fnms
      println(io, indent*indent*indent_all*"func."*string(fnm)*"=", getfield(f.func, fnm))
    end
  end
  
  println(io, indent_all*"args")
  for arg in f.args
    if istracked(arg)
      print_graph_(io, arg, indent, indent*indent_all)
    else
      println(io, indent*indent_all, arg)
    end
  end
end

"Function to print the graph of an Tracked"
function print_graph_(io::IO, x::Tracked, indent, indent_all)
  println(io, indent_all*"Tracker=")
  indent_all *= indent
  println(io, indent_all*"isleaf=", x.isleaf)
  grad = isdefined(x, :grad) ? x.grad : undef
  println(io, indent_all*"grad=", grad)
  print_graph_(io, x.f, indent, indent_all)
end

"Function to print the graph of an TrackedArray, TrackedReal, TrackedTuple}"
function print_graph(io::IO, x; indent="-")
  println(io, "TrackedData")
  println(io, indent*"data=", data(x))  
  print_graph_(io, tracker(x), indent, indent)
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
# TODO: do we need to define a rrule for nobacksies?
rrule(::typeof(nobacksies), f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
rrule(::typeof(nobacksies), f::String, x) = data(x), Δ -> error(f)
# @grad nobacksies(f::Symbol, x) = data(x), Δ -> error("Nested AD not defined for $f")
# @grad nobacksies(f::String, x) = data(x), Δ -> error(f)

param(x::Number) = TrackedReal(float(x))
param(xs::AbstractArray) = TrackedArray(float.(xs))

param(x::TrackedReal) = track(identity, x)
param(x::TrackedArray) = track(identity, x)
# TODO: do we need to define a rrule for identity?
# @grad identity(x) = data(x), Δ -> (Δ,)
rrule(::typeof(identity), x::TrackedTypes) = data(x), Δ->(NoTangent(), Δ)

# TODO: where is this code used?
import Adapt: adapt, adapt_structure

adapt_structure(T, xs::TrackedArray) = param(adapt(T, data(xs)))

end
