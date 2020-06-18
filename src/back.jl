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

function walk(f, x::Tracked; once = true)
  queue = Tracked[x]
  seen = Set{UInt64}()
  while !isempty(queue)
    x = pop!(queue)
    f(x)
    _walk(queue, seen, x.f)
    once && !x.isleaf && (x.f = Call(missing, ()))
  end
end
  
function back_(c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, d) -> back_(x, d), c.args, data.(Δs))
end

back_(::Call{Nothing}, Δ) = nothing
back_(::Call{Missing}, Δ) = error("`back!` was already used")

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
    back_(x, Δ)
    walk(x, once = once) do x
      back_(x.f, x.grad)
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
  foreach((x, Δ) -> back(g, x, Δ), c.args, Δs)
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
  walk(x, once = false) do x
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
