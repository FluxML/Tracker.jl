import Base: *, reduce
import Base: +
using ChainRules: NoTangent

import LinearAlgebra
import LinearAlgebra: inv, det, logdet, logabsdet, \, /

using Statistics
using LinearAlgebra: Diagonal, Transpose, Adjoint, diagm, diag

struct TrackedArray{T,N,A<:AbstractArray{T,N}} <: AbstractArray{T,N}
  tracker::Tracked{A}
  data::A
  grad::A
  TrackedArray{T,N,A}(t::Tracked{A}, data::A) where {T,N,A} = new(t, data)
  TrackedArray{T,N,A}(t::Tracked{A}, data::A, grad::A) where {T,N,A} = new(t, data, grad)
end

data(x::TrackedArray) = x.data
tracker(x::TrackedArray) = x.tracker

TrackedVector{T,A} = TrackedArray{T,1,A}
TrackedMatrix{T,A} = TrackedArray{T,2,A}
TrackedVecOrMat{T,A} = Union{TrackedVector{T,A},TrackedMatrix{T,A}}

track_ctor(c::Call, x::AbstractArray) = TrackedArray(c, x)

TrackedArray(c::Call, x::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c), x)

TrackedArray(c::Call, x::A, Δ::A) where A <: AbstractArray =
  TrackedArray{eltype(A),ndims(A),A}(Tracked{A}(c, Δ), x, Δ)

TrackedArray(x::AbstractArray) = TrackedArray(Call(), x, zero(x))

Base.eltype(x::Type{<:TrackedArray{T}}) where T <: Real = TrackedReal{T}

Base.convert(::Type{T}, x::S) where {T<:TrackedArray,S<:T} = x

Base.convert(T::Type{<:TrackedArray}, x::TrackedArray) =
  error("Not implemented: convert $(typeof(x)) to $T")

Base.convert(::Type{<:TrackedArray{T,N,A}}, x::AbstractArray) where {T,N,A} =
  TrackedArray(convert(A, x))

Base.show(io::IO, t::Type{TrackedArray{T,N,A}}) where {T,N,A<:AbstractArray{T,N}} =
  @isdefined(A) ?
    print(io, "TrackedArray{…,$A}") :
    invoke(show, Tuple{IO,DataType}, io, t)

function Base.summary(io::IO, x::TrackedArray)
  print(io, "Tracked ")
  summary(io, data(x))
end

Base.print_array(io::IO, x::TrackedArray) = Base.print_array(io, data(x))

function Base.show(io::IO, x::TrackedArray)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.copy(x::TrackedArray) = x

collect(xs::TrackedArray) = xs

Base.setindex!(xs::TrackedArray, v, i...; kwargs...) =
  error("Can't differentiate `setindex!`")

back!(::TrackedArray) = error("Value is not scalar; use `back!(sum(x))` or `back!(x, Δ)`")

function update!(x::TrackedArray, Δ)
  x.data .+= data(Δ)
  tracker(x).grad .= 0
  return x
end

function update!(x::AbstractArray, Δ)
  x .+= data(Δ)
  return x
end

# Fallthrough methods

for f in :[Base.size, Base.ndims, Base.collect].args
  @eval @inline $f(x::TrackedArray, a...) = $f(data(x), a...)
end

Base.size(x::TrackedArray, i::Integer, j::Integer, is::Integer...) =
  size(data(x), i, j, is...)

Base.similar(x::TrackedArray, T::Type, dims::Dims) = similar(data(x), T, dims)

for op in [:(==), :≈]
    @eval Base.$op(x::TrackedArray, y::AbstractArray) = Base.$op(data(x), y)
    @eval Base.$op(x::AbstractArray, y::TrackedArray) = Base.$op(x, data(y))
    @eval Base.$op(x::TrackedArray, y::TrackedArray) = Base.$op(data(x), data(y))
end

# Array Stdlib

Base.getindex(xs::TrackedArray, i...; kwargs...) = track(getindex, xs, i...; kwargs...)

Base.view(x::TrackedArray, inds...; kwargs...) = track(Base.view, x, inds...; kwargs...)

Base.:-(xs::TrackedArray) = track(-, xs)

Base.transpose(xs::TrackedArray) = track(transpose, xs)
Base.adjoint(xs::TrackedArray) = track(adjoint, xs)

det(xs::TrackedArray) = track(det, xs)

logdet(xs::TrackedArray) = track(logdet, xs)

logabsdet(xs::TrackedArray) = track(logabsdet, xs)

Base.repeat(xs::TrackedArray; kw...) = track(repeat, xs; kw...)

for (T, S) in [(:TrackedArray, :TrackedArray), (:TrackedArray, :AbstractArray), (:AbstractArray, :TrackedArray)]
    @eval Base.vcat(A::$T, B::$S, Cs::AbstractArray...) = track(vcat, A, B, Cs...)
    @eval Base.hcat(A::$T, B::$S, Cs::AbstractArray...) = track(hcat, A, B, Cs...)
end
for (T, S) in [(:TrackedVector, :TrackedVector), (:TrackedVector, :AbstractVector), (:AbstractVector, :TrackedVector)]
    @eval Base.vcat(A::$T, B::$S, Cs::AbstractVector...) = track(vcat, A, B, Cs...)
end
for (T, S) in [(:TrackedVecOrMat, :TrackedVecOrMat), (:TrackedVecOrMat, :AbstractVecOrMat), (:AbstractVecOrMat, :TrackedVecOrMat)]
    @eval Base.vcat(A::$T, B::$S, Cs::AbstractVecOrMat...) = track(vcat, A, B, Cs...)
    @eval Base.hcat(A::$T, B::$S, Cs::AbstractVecOrMat...) = track(hcat, A, B, Cs...)
end
for (T, S) in [(:TrackedArray, :Real), (:Real, :TrackedArray), (:TrackedArray, :TrackedArray)]
    @eval Base.vcat(A::$T, B::$S, Cs::Union{AbstractArray, Real}...) = track(vcat, A, B, Cs...)
    @eval Base.hcat(A::$T, B::$S, Cs::Union{AbstractArray, Real}...) = track(hcat, A, B, Cs...)
end
for (T, S) in [(:TrackedReal, :Real), (:Real, :TrackedReal), (:TrackedReal, :TrackedReal)]
    @eval Base.vcat(A::$T, B::$S, Cs::Real...) = track(vcat, A, B, Cs...)
    @eval Base.hcat(A::$T, B::$S, Cs::Real...) = track(hcat, A, B, Cs...)
end

Base.vcat(A::TrackedArray) = track(vcat, A)
Base.hcat(A::TrackedArray) = track(hcat, A)

Base.vcat(A::TrackedReal) = track(vcat, A)
Base.hcat(A::TrackedReal) = track(hcat, A)

for (T, S) in [(:TrackedArray, :TrackedArray), (:TrackedArray, :AbstractArray), (:AbstractArray, :TrackedArray)]
    @eval Base.cat(A::$T, B::$S, Cs::AbstractArray...; dims) = track(cat, A, B, Cs...; dims = dims)
end

Base.cat(A::TrackedArray; dims) = track(cat, A; dims = dims)

for f in [:vcat, :hcat]
  @eval function reduce(::typeof($f), xs::Vector{<:TrackedArray})
    y = reduce($f, data.(xs))
    track(Call(reduce, ($f, xs)), y)
  end
end

function scan(c::Call{typeof(reduce)})
    foreach(scan, tracker.(c.args[2]))
end

function back_(c::Call{typeof(reduce)}, Δ, once)
  f, xs = c.args
  foreach((x, d) -> back(x, d, once), tracker.(xs), data.(∇reduce(f, Δ, xs)))
end

Base.reshape(xs::TrackedArray, dims::Union{Colon,Int}...) = reshape(xs, dims)
Base.reshape(xs::TrackedArray, dims::Tuple{Vararg{Union{Int,Colon}}}) = reshape(xs, Base._reshape_uncolon(xs, dims))
Base.reshape(xs::TrackedArray, dims::Tuple{Vararg{Int}}) = track(reshape, xs, dims)


Base.permutedims(xs::TrackedArray, perm) = track(permutedims, xs, perm)

Base.PermutedDimsArray(xs::TrackedArray, perm) = track(PermutedDimsArray, xs, perm)

Base.reverse(xs::TrackedArray; dims) = track(reverse, xs, dims = dims)
Base.reverse(xs::TrackedVector) = track(reverse, xs)
Base.reverse(xs::TrackedVector, start::Integer, stop::Integer) = track(reverse, xs, start, stop)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end
_kron(a::AbstractVector, b::AbstractVector) = vec(_kron(reshape(a, :, 1), reshape(b, :, 1)))

Base.kron(a::TrackedVecOrMat, b::TrackedVecOrMat)  = _kron(a, b)
Base.kron(a::TrackedVecOrMat, b::AbstractVecOrMat) = _kron(a, b)
Base.kron(a::AbstractVecOrMat, b::TrackedVecOrMat) = _kron(a, b)


inv(A::TrackedArray) = Tracker.track(inv, A)

#       (/) rdivide
A::TrackedArray     / B::TrackedArray     = Tracker.track(/, A, B)
A::AbstractVecOrMat / B::TrackedArray     = Tracker.track(/, A, B)
A::TrackedArray     / B::AbstractVecOrMat = Tracker.track(/, A, B)

#       (\) ldivide  (left vec divide needs more work to resolve dispatch ambiguity)
A::TrackedArray     \ B::TrackedArray     = Tracker.track(\, A, B)
A::AbstractArray    \ B::TrackedArray     = Tracker.track(\, A, B)
A::TrackedArray     \ B::AbstractVecOrMat = Tracker.track(\, A, B)
A::AbstractMatrix   \ B::TrackedVecOrMat  = Tracker.track(\, A, B)
A::TrackedMatrix    \ B::TrackedVecOrMat  = Tracker.track(\, A, B)

# Reductions

Base.sum(xs::TrackedArray; dims = :) = track(sum, xs, dims = dims)
Base.sum(f::Union{Function,Type},xs::TrackedArray) = sum(f.(xs))

Base.prod(xs::TrackedArray; dims=:) = track(prod, xs; dims=dims)
Base.prod(f::Union{Function, Type}, xs::TrackedArray) = prod(f.(xs))

Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

import LinearAlgebra: dot

dot(xs::TrackedArray, ys::TrackedArray) = track(dot, xs, ys)
dot(xs::AbstractArray, ys::TrackedArray) = track(dot, xs, ys)
dot(xs::TrackedArray, ys::AbstractArray) = track(dot, xs, ys)

# TODO: still needs hacks? 
# Hacks to get std working
Statistics.std(x::TrackedArray; dims = :, mean = Statistics.mean(x, dims = dims), corrected::Bool = true) = _std(x,mean,dims,corrected)
_std(x::TrackedArray, mean, dims, corrected) = sqrt.(sum((x .- mean).^2, dims = dims) ./ (mapreduce(i -> size(x,i),*, dims) - corrected))
_std(x::TrackedArray, mean, ::Colon, corrected) = sqrt.(sum((x .- mean).^2) ./ (length(x) - corrected))

Statistics.var(x::TrackedArray; dims=:, mean=Statistics.mean(data(x); dims), corrected::Bool=true) =
  track(Statistics.var, x; dims, mean=data(mean), corrected)
# from https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Statistics/statistics.jl

LinearAlgebra.norm(x::TrackedArray{T}, p::Real = 2) where T =
  (sum(abs.(x).^p) + eps(T))^(oneunit(T) / p) # avoid d(sqrt(x))/dx == Inf at 0

Statistics.mean(xs::TrackedArray; dims = :) = track(mean, xs, dims = dims)

Base.maximum(xs::TrackedArray; dims = :) = track(maximum, xs, dims = dims)
Base.minimum(xs::TrackedArray; dims = :) = track(minimum, xs, dims = dims)

# BLAS

LinearAlgebra.diagm(x::Pair{<:Integer, <:TrackedVector}) = track(diagm, x...)

# fix Matrix(Diagonal(param([1,2,3]))) after https://github.com/JuliaLang/julia/pull/44615
(::Type{Matrix})(d::Diagonal{<:Any,<:TrackedArray}) = diagm(0 => d.diag)

x::TrackedMatrix  * y::AbstractMatrix = track(*, x, y)
x::AbstractMatrix * y::TrackedMatrix  = track(*, x, y)
x::TrackedMatrix  * y::TrackedMatrix  = track(*, x, y)

x::TrackedMatrix  * y::AbstractVector = track(*, x, y)
x::AbstractMatrix * y::TrackedVector  = track(*, x, y)
x::TrackedMatrix  * y::TrackedVector  = track(*, x, y)

x::TrackedVector  * y::AbstractVector = track(*, x, y)
x::AbstractVector * y::TrackedVector  = track(*, x, y)
x::TrackedVector  * y::TrackedVector  = track(*, x, y)

# TODO handle this
x::TrackedArray + y::TrackedArray = track(+, x, y)
x::Real * y::TrackedArray = track(*, x, y)

# Ambiguity fixes
Base.:*(x::Transpose{T,<:AbstractVector{T}},y::TrackedMatrix) where {T} = track(*, x, y)
Base.:*(x::TrackedMatrix,y::Transpose{T,<:AbstractVector{T}}) where {T} = track(*, x, y)

Base.:*(x::Transpose{T,<:AbstractMatrix{T}},y::TrackedMatrix) where {T} = track(*, x, y)
Base.:*(x::TrackedMatrix,y::Transpose{T,<:AbstractMatrix{T}}) where {T} = track(*, x, y)

Base.:*(x::Transpose{T,<:AbstractVector{T}},y::TrackedVector) where {T} = track(*, x, y)
Base.:*(x::TrackedVector,y::Transpose{T,<:AbstractVector{T}}) where {T} = track(*, x, y)

Base.:*(x::Transpose{T,<:AbstractMatrix{T}},y::TrackedVector) where {T} = track(*, x, y)
Base.:*(x::TrackedVector,y::Transpose{T,<:AbstractMatrix{T}}) where {T} = track(*, x, y)

Base.:*(x::Adjoint{T,<:AbstractVector{T}},y::TrackedMatrix) where {T} = track(*, x, y)
Base.:*(x::TrackedMatrix,y::Adjoint{T,<:AbstractVector{T}}) where {T} = track(*, x, y)

Base.:*(x::Adjoint{T,<:AbstractMatrix{T}},y::TrackedMatrix) where {T} = track(*, x, y)
Base.:*(x::TrackedMatrix,y::Adjoint{T,<:AbstractMatrix{T}}) where {T} = track(*, x, y)

Base.:*(x::Adjoint{T,<:AbstractVector{T}},y::TrackedVector) where {T} = track(*, x, y)
Base.:*(x::TrackedVector,y::Adjoint{T,<:AbstractVector{T}}) where {T} = track(*, x, y)

Base.:*(x::Adjoint{T,<:AbstractMatrix{T}},y::TrackedVector) where {T} = track(*, x, y)
Base.:*(x::TrackedVector,y::Adjoint{T,<:AbstractMatrix{T}}) where {T} = track(*, x, y)

Base.:*(x::Diagonal, y::TrackedVector) = track(*, x, y)

Base.:*(x::Diagonal, y::TrackedMatrix) = track(*, x, y)
Base.:*(x::TrackedMatrix, y::Diagonal) = track(*, x, y)

# TODO I guess we have these definitions in ChainRules
# @grad a::AbstractVecOrMat * b::AbstractVecOrMat =
#   data(a)*data(b), Δ -> (Δ * transpose(b), transpose(a) * Δ)


# Broadcasting
Base.broadcasted(f::F, x::TrackedArray, y::Union{Real, AbstractArray}) where {F} = track(Base.broadcasted, f, x, y)
Base.broadcasted(f::F, x::Union{Real, AbstractArray}, y::TrackedArray) where {F} = track(Base.broadcasted, f, x, y)
Base.broadcasted(f::F, x::TrackedArray, y::TrackedArray) where {F} = track(Base.broadcasted, f, x, y)
Base.broadcasted(f::F, x::TrackedArray) where {F} = track(Base.broadcasted, f, x) # for sin., cos. etc
# TODO: solve x.^p, where p is integer > 2; and x.^3.5
# Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::TrackedArray, val::Val_) where {Val_ <: Val{} } = Base.broadcasted(^, x, val)
Base.broadcasted(::typeof(Base.literal_pow), ::typeof(^), x::TrackedArray, ::Val{2}) = track(Base.broadcasted, Base.literal_pow, ^, x, Val(2))


# NNlib

using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, ∇conv_data, depthwiseconv, maxpool, meanpool
import NNlib: DenseConvDims, DepthwiseConvDims, PoolDims, within_gradient

within_gradient(::TrackedArray) = true
within_gradient(::TrackedReal) = true

softmax(xs::TrackedArray; dims=1) = track(softmax, xs; dims=dims)

if isdefined(NNlib, :∇softmax_data)  # use new form to avoid a depwarn, but only possible Julia 1.6+
  @eval @grad function softmax(xs; dims=1)
    y = softmax(data(xs); dims=dims)
    y, Δ -> (nobacksies(:softmax, NNlib.∇softmax_data(data(Δ), data(y); dims=dims)),)
  end
else
  @eval @grad function softmax(xs; dims=1)  # TODO delete this when dropping Julia 1.3 (and increase NNlib bound)
    y = softmax(data(xs); dims=dims)
    y, Δ -> (nobacksies(:softmax, ∇softmax(data(Δ), data(xs), data(y); dims=dims)),)
  end
end

logsoftmax(xs::TrackedArray; dims=1) = track(logsoftmax, xs; dims=dims)

if isdefined(NNlib, :∇logsoftmax_data)  # use new form to avoid a depwarn, but only possible Julia 1.6+
  @eval @grad function logsoftmax(xs; dims=1)
    y = logsoftmax(data(xs); dims=dims)
    y, Δ -> (nobacksies(:logsoftmax, NNlib.∇logsoftmax_data(data(Δ), data(y); dims=dims)),)
  end
else
  @eval @grad function logsoftmax(xs; dims=1)
    y = logsoftmax(data(xs); dims=dims)
    y, Δ -> (nobacksies(:logsoftmax, ∇logsoftmax(data(Δ), data(xs), data(y); dims=dims)),)
  end
end

depthwiseconv(x::TrackedArray, w::TrackedArray, cdims::DepthwiseConvDims; kw...) = track(depthwiseconv, x, w, cdims; kw...)
depthwiseconv(x::AbstractArray, w::TrackedArray, cdims::DepthwiseConvDims; kw...) = track(depthwiseconv, x, w, cdims; kw...)
depthwiseconv(x::TrackedArray, w::AbstractArray, cdims::DepthwiseConvDims; kw...) = track(depthwiseconv, x, w, cdims; kw...)

@grad depthwiseconv(x, w, cdims::DepthwiseConvDims; kw...) =
  depthwiseconv(data(x), data(w), cdims; kw...),
    Δ -> nobacksies(:depthwiseconv,
      (NNlib.∇depthwiseconv_data(data.((Δ, w))..., cdims; kw...),
       NNlib.∇depthwiseconv_filter(data.((x, Δ))..., cdims; kw...),
       nothing))

conv(x::TrackedArray,  w::TrackedArray, cdims::DenseConvDims;  kw...) = track(conv, x, w, cdims; kw...)
conv(x::AbstractArray, w::TrackedArray, cdims::DenseConvDims;  kw...) = track(conv, x, w, cdims; kw...)
conv(x::TrackedArray,  w::AbstractArray, cdims::DenseConvDims; kw...) = track(conv, x, w, cdims; kw...)

@grad conv(x, w, cdims::DenseConvDims; kw...) =
  conv(data(x), data(w), cdims; kw...),
    Δ -> nobacksies(:conv,
      (NNlib.∇conv_data(data.((Δ, w))..., cdims; kw...),
       NNlib.∇conv_filter(data.((x, Δ))..., cdims; kw...),
       nothing))

∇conv_data(x::TrackedArray,  w::TrackedArray, cdims::DenseConvDims;  kw...) = track(∇conv_data, x, w, cdims; kw...)
∇conv_data(x::AbstractArray, w::TrackedArray, cdims::DenseConvDims;  kw...) = track(∇conv_data, x, w, cdims; kw...)
∇conv_data(x::TrackedArray,  w::AbstractArray, cdims::DenseConvDims; kw...) = track(∇conv_data, x, w, cdims; kw...)

@grad function ∇conv_data(y, w, cdims::DenseConvDims; kw...)
  return (
    ∇conv_data(data(y), data(w), cdims; kw...),
    Δ -> begin
      return nobacksies(:conv,
        (NNlib.conv(data.((Δ, w))..., cdims; kw...),
         NNlib.∇conv_filter(data.((Δ, y))..., cdims; kw...),
         nothing)
      )
    end
  )
end

maxpool(x::TrackedArray, pdims::PoolDims; kw...) = track(maxpool, x, pdims; kw...)

@grad function maxpool(x, pdims::PoolDims; kw...)
  y = maxpool(data(x), pdims; kw...)
  y, Δ -> (nobacksies(:maxpool, NNlib.∇maxpool(data.((Δ, y, x))..., pdims; kw...)), nothing)
end

meanpool(x::TrackedArray, pdims::PoolDims; kw...) = track(meanpool, x, pdims; kw...)


@grad function meanpool(x, pdims::PoolDims; kw...)
  y = meanpool(data(x), pdims; kw...)
  y, Δ -> (nobacksies(:meanpool, NNlib.∇meanpool(data.((Δ, y, x))..., pdims; kw...)), nothing)
end