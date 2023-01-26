mutable struct TrackedReal{T<:Real} <: Real
  data::T
  tracker::Tracked{T}
end

TrackedReal(x::Real) = TrackedReal(x, Tracked{typeof(x)}(Call(), zero(x)))

data(x::TrackedReal) = x.data
tracker(x::TrackedReal) = x.tracker

ForwardDiff.value(x::TrackedReal) = x.data

track(f::Call, x::Real) = TrackedReal(x, Tracked{typeof(x)}(f, zero(x)))

function back!(x::TrackedReal; once = true)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1, once = once)
end

function update!(x::TrackedReal, Δ)
  x.data += data(Δ)
  tracker(x).grad = 0
  return x
end

function Base.show(io::IO, x::TrackedReal)
  T = get(io, :typeinfo, Any)
  show(io, data(x))
  T <: TrackedReal || print(io, " (tracked)")
end

Base.decompose(x::TrackedReal) = Base.decompose(data(x))

Base.copy(x::TrackedReal) = x

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{T}) where T = x

Base.convert(::Type{TrackedReal{T}}, x::Real) where T = TrackedReal(convert(T, x))

Base.convert(::Type{TrackedReal{T}}, x::TrackedReal{S}) where {T,S} =
  error("Not implemented: convert tracked $S to tracked $T")

(T::Type{<:TrackedReal})(x::Real) = convert(T, x)

for op in [:(==), :≈, :<, :(<=)]
  @eval Base.$op(x::TrackedReal, y::Real) = Base.$op(data(x), y)
  @eval Base.$op(x::Real, y::TrackedReal) = Base.$op(x, data(y))
  @eval Base.$op(x::TrackedReal, y::TrackedReal) = Base.$op(data(x), data(y))
end

Base.eps(x::TrackedReal) = eps(data(x))
Base.eps(::Type{TrackedReal{T}}) where T = eps(T)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedReal) = Base.$f(data(x))
end

Printf.fix_dec(x::TrackedReal, n::Int, a...) = Printf.fix_dec(data(x), n, a...)
if VERSION >= v"1.6-"
  Printf.tofloat(x::TrackedReal) = Printf.tofloat(data(x))
end

Base.float(x::TrackedReal) = x

Base.promote_rule(::Type{TrackedReal{S}},::Type{T}) where {S,T} =
  TrackedReal{promote_type(S,T)}

using Random

for f in :[rand, randn, randexp].args
  @eval Random.$f(rng::AbstractRNG,::Type{TrackedReal{T}}) where {T} = param(rand(rng,T))
end

for (M, f, arity) in DiffRules.diffrules(; filter_modules=nothing)
  if !(isdefined(@__MODULE__, M) && isdefined(getfield(@__MODULE__, M), f))
    @warn "$M.$f is not available and hence rule for it can not be defined"
    continue  # Skip rules for methods not defined in the current scope
  end
  Mf = :($M.$f)
  if arity == 1
    @eval begin
      @grad $Mf(a::Real) = $Mf(data(a)), Δ -> (Δ * $(DiffRules.diffrule(M, f, :a)),)
      $Mf(a::TrackedReal) = track($Mf, a)
    end
  elseif arity == 2
    da, db = DiffRules.diffrule(M, f, :a, :b)
    @eval begin
      @grad $Mf(a::TrackedReal, b::TrackedReal) = $Mf(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)
      @grad $Mf(a::TrackedReal, b::Real) = $Mf(data(a), b), Δ -> (Δ * $da, zero(b))
      @grad $Mf(a::Real, b::TrackedReal) = $Mf(a, data(b)), Δ -> (zero(a), Δ * $db)
      $Mf(a::TrackedReal, b::TrackedReal)  = track($Mf, a, b)
      $Mf(a::TrackedReal, b::Real) = track($Mf, a, b)
      $Mf(a::Real, b::TrackedReal) = track($Mf, a, b)
    end
  end
end

# Eliminating ambiguity
import Base:^

^(a::TrackedReal, b::Integer) = track(^, a, b)

# Tuples

struct TrackedTuple{T<:Tuple}
  data::T
  tracker::Tracked{T}
end

data(xs::TrackedTuple) = xs.data
tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)
zero_grad!(x::Tuple) = zero_grad!.(x)

track(f::Call, xs::Tuple) = TrackedTuple(xs, Tracked{typeof(xs)}(f, zero.(xs)))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

@grad function getindex(xs::TrackedTuple, i)
  data(xs)[i], Δ -> (ntuple(j -> i == j ? Δ : 0, length(xs)), nothing)
end

# Array collection

function collect(xs)
  xs = Base.collect(xs)
  track(Call(collect, (tracker.(xs),)), data.(xs))
end

function scan(c::Call{typeof(collect)})
  foreach(scan, c.args[1])
end

function back_(c::Call{typeof(collect)}, Δ, once)
  foreach((x, d) -> back(x, d, once), c.args[1], data(Δ))
end

function back_(g::Grads, c::Call{typeof(collect)}, Δ)
  foreach((x, Δ) -> back(g, x, Δ), c.args[1], Δ)
end

collectmemaybe(xs::AbstractArray{>:TrackedReal}) = collect(xs)
collectmemaybe(xs::AbstractArray{<:TrackedReal}) = collect(xs)
