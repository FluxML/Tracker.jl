struct Params
  order::Vector{Any}
  params::IdSet{Any}
  Params() = new([], IdSet())
end

@forward Params.order Base.iterate, Base.length

function Base.push!(ps::Params, x)
  if !(x in ps.params)
    push!(ps.order, x)
    push!(ps.params, x)
  end
  return ps
end

Base.push!(ps::Params, x...) = (foreach(x -> push!(ps, x), x); ps)

Params(xs) = push!(Params(), xs...)

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.order, ", ")
  print(io, "])")
end

"""
  Grads
Structure to hold gradients of various objects.

#Parameters:
  - `grads::IdDict{Any,Any}`: dictionary to map the objects to their gradients
"""
struct Grads
  grads::IdDict{Any,Any}
end

Base.show(io::IO, ps::Grads) = println(io, "Grads(...)")

Grads() = Grads(IdDict())

@forward Grads.grads Base.setindex!, Base.haskey, Base.length, Base.iterate

"""
For each parameter `p`` in `ps`, creates a tracked gradient (of type `TrackedArray` etc.), 
because we are going to record the operations on it.
`tracker(p)` is mapped to this gradient via an  `IdDict`.
"""
Grads(ps::Params) = Grads(IdDict(tracker(p) => init_grad(data(p)) for p in ps))

Base.getindex(g::Grads, x::Tracked) = g.grads[x]

function Base.getindex(g::Grads, x)
  istracked(x) || error("Object not tracked: $x")
  g[tracker(x)]
end

accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] .+ Δ : Δ
