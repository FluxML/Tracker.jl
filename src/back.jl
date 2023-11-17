# The AD generates fairly large backtraces that are unhelpful if you interrupt
# while training; this just cleans that up.
macro interrupts(ex)
  :(try $(esc(ex))
    catch e
      e isa InterruptException || rethrow()
      throw(e)
    end)
end

# In-place gradients

init_grad(x) = zero(x)
zero_grad!(x) = zero(x)
zero_grad!(x::AbstractArray) = (x .= 0)

function _walk(queue, seen, c::Call)
  foreach(c.args) do x
    x === nothing && return
    id = objectid(x)
    if id ∉ seen
      push!(seen, id)
      pushfirst!(queue, x)
    end
    return
  end
end

function walk(f, x::Tracked, seen = Set{UInt64}(); once = true)
  queue = Tracked[x]
  while !isempty(queue)
    x = pop!(queue)
    f(x, seen)
    _walk(queue, seen, x.f)
    once && !x.isleaf && (x.f = Call(missing, ()))
  end
end
  
function back_(c::Call, Δ, seen)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach(c.args) do x
    isdefined(x, :grad) || return
    objectid(x) ∉ seen && zero_grad!(x.grad)
  end
  foreach((x, d) -> back_(x, d), c.args, data.(Δs))
end

back_(::Call{Nothing}, Δ, seen) = nothing
back_(::Call{Missing}, Δ, seen) = error("`back!` was already used")

accum!(x, Δ) = x .+ Δ
accum!(x::AbstractArray, Δ) = (x .+= Δ)

function back_(x::Tracked, Δ)
    if isdefined(x, :grad)
      x.grad = accum!(x.grad, Δ)
    else
      x.grad = Δ
    end
    return
end

back_(::Nothing, Δ) = return

function back(x::Tracked, Δ, once)
    seen = Set{UInt64}(objectid(x))
    if isdefined(x, :grad)
      x.grad = zero_grad!(x.grad)
    end
    back_(x, Δ)
    walk(x, seen, once = once) do x, seen
      back_(x.f, x.grad, seen)
    end
end

back(::Nothing, Δ, once) = return

# Interface methods

# TODO: if an error occurs in `back` the refcounts will be broken
# and `back` will silently fail to update.
# (but only if you re-use intermediate values between passes)
# Refcounts are also probably not safe in some situations (e.g. back called
# from within a backpropagator)

function back!(x, Δ; once = true)
  istracked(x) || return
  back(tracker(x), Δ, once)
  return
end

function extract_grad!(x)
  x̄ = copy(grad(x))
  x̄ = nobacksies("Use `gradient(...; nest = true)` for nested derivatives", x̄)
  tracker(x).grad = zero_grad!(grad(x))
  return x̄
end

function gradient_(f, xs...)
  xs = param.(data.(xs))
  l = f(xs...)
  losscheck(l)
  @interrupts back!(l)
  extract_grad!.(xs)
end

function gradient_(f, xs::Params)
  l = f()
  losscheck(l)
  @interrupts back!(l)
  gs = Grads()
  for x in xs
    gs[tracker(x)] = extract_grad!(x)
  end
  return gs
end

# Out-of-place gradients

function back_(g::Grads, c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, Δ) -> back_(g, x, Δ), c.args, Δs)
end

back_(g::Grads, ::Call{Nothing}, Δ) = nothing

function back_(g::Grads, x::Tracked, Δ)
  x.isleaf && (accum!(g, x, Δ); return)
  accum!(g, x, Δ)
  return
end

back_(g::Grads, ::Nothing, Δ) = return

function back(g::Grads, x::Tracked, Δ)
  back_(g, x, Δ)
  walk(x, once = false) do x, seen
    back_(g, x.f, g[x])
  end
end

back(::Grads, ::Nothing, _) = return

collectmemaybe(xs) = xs

function forward(f, ps::Params)
  y = collectmemaybe(f())
  y, function (Δ)
    g = Grads(ps)
    if istracked(y)
      back(g, tracker(y), Δ)
    end
    return g
  end
end

function forward(f, args...)
  args = param.(args)
  y, back = forward(() -> f(args...), Params(args))
  y, Δ -> getindex.(Ref(back(Δ)), args)
end

function losscheck(x)
  x isa Real || error("Function output is not scalar")
  isinf(x) && error("Loss is infinite")
  isnan(x) && error("Loss is NaN")
end

function gradient_nested(f, args...)
  y, back = forward(f, args...)
  losscheck(y)
  return back(1)
end

gradient(f, xs...; nest = false) =
  nest ? gradient_nested(f, xs...) : gradient_(f, xs...)

# Jacobians and Hessians

"""
    J = jacobian(m,x)
    
Calculate the output jacobian `J = d/dx m(x)` such that each row `i` of `J` corresponds to the gradient `J[i,:] = ∇ₓ(m(x)[i])`
"""
function jacobian(f, x::AbstractVector)
  y::AbstractVector, back = forward(f, x)
  ȳ(i) = [i == j for j = 1:length(y)]
  vcat([transpose(back(ȳ(i))[1]) for i = 1:length(y)]...)
end

hessian(f, x) = jacobian(x -> gradient(f, x, nest=true)[1], x)

using Functors: fmap, functor
using Optimisers: _trainable, isnumeric

"""
    withgradient(f, xs...)

This computes the value `f(xs...)` and the gradient with respect to `xs`.
However, it differs from `gradient` in several other respects:
* It will recurse into `xs` using `fmap`, and thus like Zygote's "explicit mode" it
  returns a tree-like gradient matching the shape of a Flux model.
  This recursion obeys restrictions imposed by `Optimisers.trainable`, if defined.
* Only objects satisfying `Optimisers.isnumeric` are regarded as parameters,
  thus in particular integers are ignored.
* Returns plain arrays, not tracked. Uses `nothing` as a strong zero gradient, like Zygote.

# Examples
```
julia> nt = (vec = [1.0, 2.0], mat = [4.0;;], fun = sin);

julia> withgradient(nt, 2) do x, p
         sum(abs2, x.vec) ^ p
       end
(val = 25.0, grad = ((vec = [20.0, 40.0], mat = [0.0;;], fun = nothing), nothing))

julia> using Flux

julia> model = Chain(Dense(2 => 1, tanh), Dense(1 => 1, bias=false));

julia> withgradient(model, rand(Float32, 2)) do m, x
         sum(abs2, m(x))
       end
(val = 0.035716165f0, grad = ((layers = ((weight = Float32[-0.4241869 -0.16741231], bias = Float32[-0.5529184], σ = nothing), (weight = Float32[-0.04804218;;], bias = nothing, σ = nothing)),), Float32[0.12706584, -0.08858479]))
```
"""
function withgradient(f, xs...)
    pxs = fmap(param, xs; exclude = isnumeric, walk = _trainable_walk)
    l = f(pxs...)
    losscheck(l)
    l isa TrackedReal || return (val = l, grad = nothing)
    @interrupts back!(l)
    (val = data(l), grad = rec_grad(pxs))
end

function _trainable_walk(f, x)
  func, re = functor(x)
  isempty(func) && return x
  done = map(f, _trainable(x))  # recurse only into trainable fields, this contains `nothing` elsewhere
  map(func, merge(func, done)) do n, t
      isnothing(t) ? n : t
  end |> re  # reconstruct the whole thing
end
_trainable_walk(f, x::Tuple) = map(f, x)

# Easier to write the recursion to extract the gradients without using fmap:
rec_grad(x::TrackedArray) = grad(x)
rec_grad(x::TrackedReal) = grad(x)
rec_grad(x::AbstractArray{<:Number}) = nothing
rec_grad(x::Number) = nothing

rec_grad(x::Union{Tuple,NamedTuple,AbstractArray}) = map(rec_grad, x)
rec_grad(::Tuple{}) = nothing
rec_grad(::NamedTuple{(), Tuple{}}) = nothing
function rec_grad(x::T) where {T}
    F = fieldnames(T)
    isempty(F) && return nothing
    map(f -> rec_grad(getfield(x, f)), NamedTuple{F}(F))
end
