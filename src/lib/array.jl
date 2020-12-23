import Base: *, reduce

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

track(c::Call, x::AbstractArray) = TrackedArray(c, x)

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

@grad function getindex(xs::AbstractArray, i...; kwargs...)
  getindex(data(xs), i...; kwargs...), function (Δ)
        Δ′ = zero(xs)
        setindex!(Δ′, data(Δ), i...; kwargs...)
        (nobacksies(:getindex, Δ′), map(_->nothing, i)...)
    end
end

@grad function getindex(xs::AbstractArray, i::Array...)
  data(xs)[i...], function (Δ)
    Δ′ = zero(xs)
    @views Δ′[i...] .+= data(Δ)
    (nobacksies(:getindex, Δ′), map(_->nothing, i)...)
  end
end

Base.view(x::TrackedArray, inds...; kwargs...) = track(Base.view, x, inds...; kwargs...)

@grad function view(x::AbstractArray, inds...; kwargs...)
    view(data(x), inds...; kwargs...), function (Δ)
        grad_output = zero(x)
        subgrad = view(grad_output, inds...; kwargs...)
        subgrad[:] = data(Δ)
        (nobacksies(:view, grad_output), map(_->nothing, inds)...)
    end
end

Base.:-(xs::TrackedArray) = track(-, xs)

@grad -(xs) = -data(xs), Δ -> (-Δ,)

Base.transpose(xs::TrackedArray) = track(transpose, xs)
Base.adjoint(xs::TrackedArray) = track(adjoint, xs)

@grad transpose(xs) = transpose(data(xs)), Δ -> (trim(xs, transpose(Δ)),)
@grad adjoint(xs) = data(xs)', Δ -> (trim(xs, Δ'),)

det(xs::TrackedArray) = track(det, xs)
@grad det(xs) = det(data(xs)), Δ -> (Δ * det(xs) * transpose(inv(xs)),)

logdet(xs::TrackedArray) = track(logdet, xs)
@grad logdet(xs) = logdet(data(xs)), Δ -> (Δ * transpose(inv(xs)),)

logabsdet(xs::TrackedArray) = track(logabsdet, xs)
@grad logabsdet(xs) = logabsdet(data(xs)), Δ -> (Δ[1] * transpose(inv(xs)),)

Base.repeat(xs::TrackedArray; kw...) = track(repeat, xs; kw...)

@grad function repeat(xs; inner=ntuple(x->1, ndims(xs)), outer=ntuple(x->1, ndims(xs)))
  repeat(data(xs), inner = inner, outer = outer), function (Δ)
    Δ′ = zero(xs)
    S = size(xs)

    # Loop through each element of Δ, calculate source dimensions, accumulate into Δ′
    for (dest_idx, val) in pairs(IndexCartesian(), data(Δ))
        # First, round dest_idx[dim] to nearest gridpoint defined by inner[dim], then
        # wrap around based on original size S.
        src_idx = [mod1(div(dest_idx[dim] - 1, inner[dim]) + 1, S[dim]) for dim in 1:length(S)]
        Δ′[src_idx...] += val
    end
    (nobacksies(:repeat, Δ′),)
  end
end

function combinations(xs, n)
  n < 1 && return [[]]
  cs = combinations(xs, n-1)
  [[x, c...] for x in xs, c in cs]
end

for i = 0:2, c = combinations([:AbstractArray, :TrackedArray, :Number], i), f = [:hcat, :vcat]
  cnames = map(_ -> gensym(), c)
  @eval Base.$f($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::Union{TrackedArray,TrackedReal}, xs::Union{AbstractArray,Number}...) =
    track($f, $(cnames...), x, xs...)
end

for i = 0:2, c = combinations([:AbstractVecOrMat, :TrackedVecOrMat], i), f = [:hcat, :vcat]
  cnames = map(_ -> gensym(), c)
  @eval Base.$f($([:($x::$c{T}) for (x, c) in zip(cnames, c)]...), x::TrackedVecOrMat{T}, xs::AbstractVecOrMat{T}...) where T =
    track($f, $(cnames...), x, xs...)
end

for i = 0:2, c = combinations([:AbstractVector, :TrackedVector], i), f = [:hcat, :vcat]
  cnames = map(_ -> gensym(), c)
  @eval Base.$f($([:($x::$c{T}) for (x, c) in zip(cnames, c)]...), x::TrackedVector{T}, xs::AbstractVector{T}...) where T =
    track($f, $(cnames...), x, xs...)
end

@grad function vcat(xs...)
  vcat(data.(xs)...), function (Δ)
    start = 0
    Δs = [begin
      x = map(_ -> :, size(xsi))
      i = isempty(x) ? x : Base.tail(x)
      d = Δ[start+1:start+size(xsi,1), i...]
      start += size(xsi, 1)
      d
    end for xsi in xs]
    return (Δs...,)
  end
end

@grad function hcat(xs...)
  hcat(data.(xs)...), function (Δ)
    start = 0
    Δs = [begin
      d = if ndims(xsi) == 1
        Δ[:, start+1]
      else
        i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
        Δ[:, start+1:start+size(xsi,2), i...]
      end
      start += size(xsi, 2)
      d
    end for xsi in xs]
    return (Δs...,)
  end
end

for i = 0:2, c = combinations([:AbstractArray, :TrackedArray], i)
  cnames = map(_ -> gensym(), c)
  @eval Base.cat($([:($x::$c) for (x, c) in zip(cnames, c)]...), x::TrackedArray, xs::AbstractArray...; dims) =
    track(cat, $(cnames...), x, xs..., dims = dims)
end

@grad function cat(Xs...; dims)
  cat(data.(Xs)..., dims = dims), function (Δ)
    start = ntuple(i -> 0, Val(ndims(Δ)))
    Δs = [begin
      dim_xs = 1:ndims(xs)
      till_xs = ntuple((i -> i in dims ? (i in dim_xs ? size(xs,i) : 1) : 0), Val(ndims(Δ)))
      xs_in_Δ = ntuple(i -> till_xs[i] > 0 ? (start[i]+1:start[i]+till_xs[i]) : Colon(), Val(ndims(Δ)))
      d = reshape(Δ[xs_in_Δ...],size(xs))
      start = start .+ till_xs
      d
    end for xs in Xs]
    return (Δs...,)
  end
end

function ∇reduce(f::typeof(vcat), Δ, xs)
  start = 0
  Δs = [begin
    x = map(_ -> :, size(xsi))
    i = isempty(x) ? x : Base.tail(x)
    d = Δ[start+1:start+size(xsi,1), i...]
    start += size(xsi, 1)
    d
  end for xsi in xs]
  return Δs
end

function ∇reduce(f::typeof(hcat), Δ, xs)
  start = 0
  Δs = [begin
    d = if ndims(xsi) == 1
      Δ[:, start+1]
    else
      i = map(_ -> :, size(xsi)) |> Base.tail |> Base.tail
      Δ[:, start+1:start+size(xsi,2), i...]
    end
    start += size(xsi, 2)
    d
  end for xsi in xs]
  return Δs
end

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

@grad reshape(xs, dims) = reshape(data(xs), dims), Δ -> (reshape(Δ, size(xs)),nothing)

Base.permutedims(xs::TrackedArray, perm) = track(permutedims, xs, perm)
@grad permutedims(xs, perm) = permutedims(data(xs), perm), Δ -> (permutedims(Δ, invperm(perm)),nothing)

Base.PermutedDimsArray(xs::TrackedArray, perm) = track(PermutedDimsArray, xs, perm)
@grad PermutedDimsArray(xs, perm) = PermutedDimsArray(data(xs), perm), Δ -> (PermutedDimsArray(Δ, invperm(perm)),nothing)

Base.reverse(xs::TrackedArray; dims) = track(reverse, xs, dims = dims)
@grad reverse(xs; dims) = reverse(data(xs), dims = dims), Δ -> (reverse(Δ, dims = dims), nothing)
Base.reverse(xs::TrackedVector) = track(reverse, xs)
@grad reverse(xs::TrackedVector) = reverse(data(xs)), Δ -> (reverse(Δ),)
Base.reverse(xs::TrackedVector, start, stop) = track(reverse, xs, start, stop)
@grad reverse(xs, start, stop) = reverse(data(xs), start, stop), Δ -> (reverse(Δ, start, stop), nothing, nothing)

function _kron(mat1::AbstractMatrix,mat2::AbstractMatrix)
    m1, n1 = size(mat1)
    mat1_rsh = reshape(mat1,(1,m1,1,n1))

    m2, n2 = size(mat2)
    mat2_rsh = reshape(mat2,(m2,1,n2,1))

    return reshape(mat1_rsh.*mat2_rsh, (m1*m2,n1*n2))
end

Base.kron(a::TrackedMatrix, b::TrackedMatrix)  = _kron(a, b)
Base.kron(a::TrackedMatrix, b::AbstractMatrix) = _kron(a, b)
Base.kron(a::AbstractMatrix, b::TrackedMatrix) = _kron(a, b)


inv(A::TrackedArray) = Tracker.track(inv, A)
@grad function inv(A)
    return inv(Tracker.data(A)), function (Δ)
        Ainv = inv(A)
        ∇A = - Ainv' * Δ * Ainv'
        return (∇A, )
    end
end

#       (/) rdivide
A::TrackedArray     / B::TrackedArray     = Tracker.track(/, A, B)
A::AbstractVecOrMat / B::TrackedArray     = Tracker.track(/, A, B)
A::TrackedArray     / B::AbstractVecOrMat = Tracker.track(/, A, B)
@grad function Base.:/(A, B)
    return Tracker.data(A) / Tracker.data(B), function (Δ)
        ∇A = Δ / B'
        ∇B = - (A / B)' * ∇A
        return (∇A, ∇B)
    end
end

#       (\) ldivide  (left vec divide needs more work to resolve dispatch ambiguity)
A::TrackedArray     \ B::TrackedArray     = Tracker.track(\, A, B)
A::AbstractArray    \ B::TrackedArray     = Tracker.track(\, A, B)
A::TrackedArray     \ B::AbstractVecOrMat = Tracker.track(\, A, B)
A::AbstractMatrix   \ B::TrackedVecOrMat  = Tracker.track(\, A, B)
A::TrackedMatrix    \ B::TrackedVecOrMat  = Tracker.track(\, A, B)
@grad function Base.:\(A, B)
    return Tracker.data(A) \ Tracker.data(B), function (Δ)
        ∇B = A' \ Δ
        ∇A = - ∇B * (A \ B)'
        return (∇A, ∇B)
    end
end


# Reductions

Base.sum(xs::TrackedArray; dims = :) = track(sum, xs, dims = dims)
Base.sum(f::Union{Function,Type},xs::TrackedArray) = sum(f.(xs))

@grad sum(xs; dims = :) = sum(data(xs), dims = dims),
  Δ -> (zero(xs) .+ Δ, )

Base.prod(xs::TrackedArray; dims=:) = track(prod, xs; dims=dims)
Base.prod(f::Union{Function, Type}, xs::TrackedArray) = prod(f.(xs))

@grad function prod(xs; dims=:)
  p = prod(data(xs); dims=dims)
  p, Δ -> (p ./ xs .* Δ,)
end

Base.findfirst(xs::TrackedArray, args...) = findfirst(xs.data, args...)

Statistics.mean(xs::TrackedArray; dims = :) = track(mean, xs, dims = dims)

Base.maximum(xs::TrackedArray; dims = :) = track(maximum, xs, dims = dims)
Base.minimum(xs::TrackedArray; dims = :) = track(minimum, xs, dims = dims)

import LinearAlgebra: dot

dot(xs::TrackedArray, ys::TrackedArray) = track(dot, xs, ys)
dot(xs::AbstractArray, ys::TrackedArray) = track(dot, xs, ys)
dot(xs::TrackedArray, ys::AbstractArray) = track(dot, xs, ys)

@grad dot(xs, ys) = dot(data(xs), data(ys)), Δ -> (Δ .* ys, Δ .* xs)

# Hacks to get std working
Statistics.std(x::TrackedArray; dims = :, mean = Statistics.mean(x, dims = dims), corrected::Bool = true) = _std(x,mean,dims,corrected)
_std(x::TrackedArray, mean, dims, corrected) = sqrt.(sum((x .- mean).^2, dims = dims) ./ (mapreduce(i -> size(x,i),*, dims) - corrected))
_std(x::TrackedArray, mean, ::Colon, corrected) = sqrt.(sum((x .- mean).^2) ./ (length(x) - corrected))

LinearAlgebra.norm(x::TrackedArray{T}, p::Real = 2) where T =
  (sum(abs.(x).^p) + eps(T))^(oneunit(T) / p) # avoid d(sqrt(x))/dx == Inf at 0

@grad mean(xs; dims = :) = mean(data(xs), dims=dims), Δ -> (_backmean(xs,Δ,dims),)
_backmean(xs, Δ, ::Colon) = zero(xs) .+ Δ ./ length(xs)
_backmean(xs, Δ, dims) = zero(xs) .+ Δ ./ mapreduce(i -> size(data(xs),i),*,dims)

@grad function maximum(xs; dims = dims)
  maximum(data(xs), dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmax(data(xs), dims = dims)
    Δ′[i] = data(Δ)
    return (nobacksies(:maximum, Δ′),)
  end
end

@grad function minimum(xs;  dims = dims)
  minimum(data(xs),  dims = dims), function (Δ)
    Δ′ = zero(xs)
    _, i = findmin(data(xs),  dims = dims)
    Δ′[i] = data(Δ)
    return (nobacksies(:minimum, Δ′),)
  end
end

# BLAS

LinearAlgebra.diagm(x::Pair{<:Integer, <:TrackedVector}) = track(diagm, x...)
@grad diagm(i, x) = diagm(i => data(x)), Δ -> (nothing, diag(Δ, i))

x::TrackedMatrix  * y::AbstractMatrix = track(*, x, y)
x::AbstractMatrix * y::TrackedMatrix  = track(*, x, y)
x::TrackedMatrix  * y::TrackedMatrix  = track(*, x, y)

x::TrackedMatrix  * y::AbstractVector = track(*, x, y)
x::AbstractMatrix * y::TrackedVector  = track(*, x, y)
x::TrackedMatrix  * y::TrackedVector  = track(*, x, y)

x::TrackedVector  * y::AbstractVector = track(*, x, y)
x::AbstractVector * y::TrackedVector  = track(*, x, y)
x::TrackedVector  * y::TrackedVector  = track(*, x, y)

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

@grad a::AbstractVecOrMat * b::AbstractVecOrMat =
  data(a)*data(b), Δ -> (Δ * transpose(b), transpose(a) * Δ)

# NNlib

using NNlib
import NNlib: softmax, ∇softmax, logsoftmax, ∇logsoftmax, conv, ∇conv_data, depthwiseconv, maxpool, meanpool
import NNlib: DenseConvDims, DepthwiseConvDims, PoolDims

softmax(xs::TrackedArray; dims=1) = track(softmax, xs; dims=dims)

@grad softmax(xs; dims=1) = softmax(data(xs); dims=dims), Δ -> (nobacksies(:softmax, ∇softmax(data(Δ), data(xs); dims=dims)),)

logsoftmax(xs::TrackedArray; dims=1) = track(logsoftmax, xs; dims=dims)

@grad logsoftmax(xs; dims=1) = logsoftmax(data(xs); dims=dims), Δ -> (nobacksies(:logsoftmax, ∇logsoftmax(data(Δ), data(xs); dims=dims)),)

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

# Broadcasting

using ForwardDiff: Dual, partials, value

trim(x, Δ) = reshape(Δ, ntuple(i -> size(Δ, i), Val(ndims(x))))

unbroadcast(x::AbstractArray, Δ) =
  size(x) == size(Δ) ? Δ :
  length(x) == length(Δ) ? trim(x, Δ) :
    trim(x, sum(Δ, dims = ntuple(i -> size(x, i) == 1 ? i : ndims(Δ)+1, Val(ndims(Δ)))))

unbroadcast(x::Number, Δ) = sum(Δ)
unbroadcast(x::Base.RefValue, _) = nothing

dual(x, p) = x
dual(x::Real, p) = Dual(x, p)

function partial(f::F, Δ, i, args::Vararg{Any,N}) where {F,N}
  dargs = ntuple(j -> dual(args[j], i==j), Val(N))
  return Δ * f(dargs...).partials[1]
end

@inline function ∇broadcast(f::F, args::Vararg{Any,N}) where {F,N}
  y = broadcast(f, data.(args)...)
  (eltype(y) <: Real && eltype(y) !== Bool) || return y
  function back(Δ)
    Δargs = ntuple(i -> partial.(f, Δ, i, args...), Val(N))
    dxs = map(unbroadcast, args, Δargs)
    return dxs
  end
  # So we can return non-tracked arrays
  track(Call(back, tracker.(args)), y)
end

using Base.Broadcast: BroadcastStyle, ArrayStyle, Broadcasted, broadcasted

struct TrackedStyle <: BroadcastStyle end

Broadcast.BroadcastStyle(::Type{<:Union{TrackedArray,TrackedReal}}) = TrackedStyle()
Broadcast.BroadcastStyle(::TrackedStyle, ::BroadcastStyle) = TrackedStyle()

# We have to re-build the original broadcast struct to get the appropriate array
# style. We need this primarily to support CuArrays' broadcasting fixes.
broadcast_rebuild(xs) = data(xs)

broadcast_rebuild(bc::Broadcasted) =
  Broadcasted(bc.f, map(broadcast_rebuild, bc.args))

preprocess(x) = x

function Base.copy(bc::Broadcasted{TrackedStyle})
  bc1 = Broadcast.flatten(bc)
  bc2 = Broadcast.flatten(broadcast_rebuild(bc))
  ∇broadcast(bc2.f, bc1.args...)
end

using Requires

# https://github.com/FluxML/Flux.jl/issues/353
if VERSION < v"1.1.0-DEV.548"
  @init Requires.isprecompiling() || @eval Base.Broadcast begin
    function flatten(bc::Broadcasted{Style}) where {Style}
      isflat(bc) && return bc
      args = cat_nested(bc)
      let makeargs = make_makeargs(bc), f = bc.f
        newf = @inline function(args::Vararg{Any,N}) where N
          f(makeargs(args...)...)
        end
        return Broadcasted{Style}(newf, args, bc.axes)
      end
    end
    @inline function make_makeargs(makeargs, t::Tuple{<:Broadcasted,Vararg{Any}})
      bc = t[1]
      let makeargs = make_makeargs(makeargs, tail(t)), f = bc.f
        let makeargs = make_makeargs(makeargs, bc.args)
          headargs, tailargs = make_headargs(bc.args), make_tailargs(bc.args)
          return @inline function(args::Vararg{Any,N}) where N
            args1 = makeargs(args...)
            a, b = headargs(args1...), tailargs(args1...)
            (f(a...), b...)
          end
        end
      end
    end
  end
end
