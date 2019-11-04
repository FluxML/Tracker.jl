mutable struct TrackedComplex{T<:Complex} <: Real
  data::T
  tracker::Tracked{T}
end

TrackedComplex(x::Complex) = TrackedComplex(x, Tracked{typeof(x)}(Call(), zero(x)))
TrackedComplex(x::Real) = TrackedComplex(x, Tracked{typeof(x)}(Call(), zero(x)))

data(x::TrackedComplex) = x.data
tracker(x::TrackedComplex) = x.tracker

track(f::Call, x::Complex) = TrackedComplex(x, Tracked{typeof(x)}(f, zero(x)))

function back!(x::TrackedComplex; once = true)
    isinf(x) && error("Loss is Inf")
    isnan(x) && error("Loss is NaN")
    return back!(x, 1, once = once)
end

function update!(x::TrackedComplex, Δ)
  x.data += data(Δ)
  tracker(x).grad = 0
  return x
end

function Base.show(io::IO, x::TrackedComplex)
  T = get(io, :typeinfo, Any)
  show(io, data(x))
  T <: TrackedComplex || print(io, " (tracked)")
end

Base.decompose(x::TrackedComplex) = Base.decompose(data(x))

Base.copy(x::TrackedComplex) = x

Base.convert(::Type{TrackedComplex{T}}, x::TrackedComplex{T}) where T = x

Base.convert(::Type{TrackedComplex{T}}, x::Complex) where T = TrackedComplex(convert(T, x))
Base.convert(::Type{TrackedComplex{T}}, x::Real) where T = TrackedComplex(convert(T, x))

Base.convert(::Type{TrackedComplex{T}}, x::TrackedComplex{S}) where {T,S} =
  error("Not implemented: convert tracked $S to tracked $T")

(T::Type{<:TrackedComplex})(x::Complex) = convert(T, x)

for op in [:(==), :≈, :<, :(<=)]
  @eval Base.$op(x::TrackedComplex, y::Complex) = Base.$op(data(x), y)
  @eval Base.$op(x::Complex, y::TrackedComplex) = Base.$op(x, data(y))
  @eval Base.$op(x::TrackedComplex, y::TrackedComplex) = Base.$op(data(x), data(y))
end

Base.eps(x::TrackedComplex) = eps(data(x))
Base.eps(::Type{TrackedComplex{T}}) where T = eps(T)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedComplex) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedComplex, n::Int, a...) = Base.Printf.fix_dec(data(x), n, a...)

Base.float(x::TrackedComplex) = x

Base.promote_rule(::Type{TrackedComplex{S}},::Type{T}) where {S,T} =
  TrackedComplex{promote_type(S,T)}

using Random

for f in :[rand, randn, randexp].args
  @eval Random.$f(rng::AbstractRNG,::Type{TrackedComplex{T}}) where {T} = param(rand(rng,T))
end

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    @grad $M.$f(a::Complex) =
      $M.$f(data(a)), Δ -> (Δ * $(DiffRules.diffrule(M, f, :a)),)
    $M.$f(a::TrackedComplex) = track($M.$f, a)
  end
end


for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :a, :b)
  f = :($M.$f)
  @eval begin
    @grad $f(a::TrackedComplex, b::TrackedComplex) = $f(data(a), data(b)), Δ -> (Δ * $da, Δ * $db)

    @grad $f(a::TrackedComplex, b::Complex) = $f(data(a), b), Δ -> (Δ * $da, _zero(b))
    @grad $f(a::TrackedComplex, b::Real) = $f(data(a), b), Δ -> (Δ * $da, _zero(b))

    @grad $f(a::Complex, b::TrackedComplex) = $f(a, data(b)), Δ -> (_zero(a), Δ * $db)
    @grad $f(a::Real, b::TrackedComplex) = $f(a, data(b)), Δ -> (_zero(a), Δ * $db)

    $f(a::TrackedComplex, b::TrackedComplex)  = track($f, a, b)

    $f(a::TrackedComplex, b::Complex) = track($f, a, b)
    $f(a::TrackedComplex, b::Real) = track($f, a, b)

    $f(a::Complex, b::TrackedComplex) = track($f, a, b)
    $f(a::Real, b::TrackedComplex) = track($f, a, b)
  end
end

using ForwardDiff: Dual
import Base:^

^(a::TrackedComplex, b::Integer) = track(^, a, b)
(T::Type{<:Complex})(x::Dual) = Dual(T(x.value), map(T, x.partials.values))