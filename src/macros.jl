"""
    @grad_from_chainrules f(args...; kwargs...)

The `@grad_from_chainrules` macro provides a way to import adjoints(rrule) defined in
ChainRules to Tracker. One must provide a method signature to import the corresponding
rrule. In the provided method signature, one should replace the types of arguments to which
one wants to take derivatives with respect with Tracker.TrackedReal and Tracker.TrackedArray
respectively. For example, we can import rrule of `f(x::Real, y::Array)`` like below:

    Tracker.@grad_from_chainrules f(x::TrackedReal, y::TrackedArray)
    Tracker.@grad_from_chainrules f(x::TrackedReal, y::Array)
    Tracker.@grad_from_chainrules f(x::Real, y::TrackedArray)

Acceptable type annotations are `TrackedReal`, `TrackedArray`, `TrackedVector`, and
`TrackedMatrix`. These can have parameters like `TrackedArray{Float32}`.
"""
macro grad_from_chainrules(fcall)
  @assert isdefined(__module__, :Tracker) "Tracker not found in module $__module__. Please load `Tracker.jl`."
  Meta.isexpr(fcall, :call) && length(fcall.args) ≥ 2 ||
    error("`@grad_from_chainrules` has to be applied to a function signature")

  f = fcall.args[1]
  # Check if kwargs... splatting is present
  kws_var = Meta.isexpr(fcall.args[2], :parameters) ? fcall.args[2].args[1].args[1] :
            nothing
  rem_args = Meta.isexpr(fcall.args[2], :parameters) ? fcall.args[3:end] :
             fcall.args[2:end]
  xs = map(rem_args) do x
    Meta.isexpr(x, :(::)) || return x
    length(x.args) == 1 && return :($(gensym())::$(x.args[1])) # ::T without var name
    @assert length(x.args) == 2
    return :($(x.args[1])::$(x.args[2])) # x::T
  end
  xs_untyped = map(xs) do x
    Meta.isexpr(x, :(::)) || return x
    return x.args[1]
  end

  untrack_args = map(enumerate(xs)) do (i, x)
    Meta.isexpr(x, :(::)) || return (x, nothing)
    name, type = x.args
    type = __strip_type(type)
    type in (:TrackedArray, :TrackedVector, :TrackedMatrix, :TrackedReal) || return (name, nothing)
    xdata = gensym(name)
    return xdata, :($(xdata) = $(Tracker.data)($(name)))
  end
  untrack_calls = filter(Base.Fix2(!==, nothing), last.(untrack_args))
  @assert length(untrack_calls) > 0 "No tracked arguments found."
  var_names = first.(untrack_args)

  f_sym = Meta.quot(Symbol(f))

  if kws_var === nothing
    return esc(quote
      $(f)($(xs...)) = $(Tracker.track)($(f), $(xs_untyped...))
      function Tracker._forward(::typeof($(f)), $(xs...))
        $(untrack_calls...)
        y, pb_f = $(CRC.rrule)($(f), $(var_names...))
        ∇internal_generated = let pb_f = pb_f # Avoid Boxing
          Δ -> return Tracker.nobacksies($(f_sym), $(__no_crctangent).(pb_f($(data)(Δ))[2:end]))
        end
        return y, ∇internal_generated
      end
    end)
  end
  return esc(quote
    function $(f)($(xs...); $(kws_var)...)
      return Tracker.track($(f), $(xs_untyped...); $(kws_var)...)
    end
    function Tracker._forward(::typeof($(f)), $(xs...); $(kws_var)...)
      $(untrack_calls...)
      y, pb_f = $(CRC.rrule)($(f), $(var_names...); $(kws_var)...)
      ∇internal_generated = let pb_f = pb_f # Avoid Boxing
        Δ -> Tracker.nobacksies($(f_sym), $(__no_crctangent).(pb_f($(data)(Δ))[2:end]))
      end
      return y, ∇internal_generated
    end
  end)
end

@inline __no_crctangent(::CRC.NoTangent) = nothing
@inline __no_crctangent(::CRC.ZeroTangent) = nothing
@inline __no_crctangent(x::CRC.AbstractThunk) = CRC.unthunk(x)
@inline __no_crctangent(x) = x

@inline function __strip_type(type)
  Meta.isexpr(type, :curly) && (type = type.args[1]) # Strip parameters from types
  Meta.isexpr(type, :(.)) && (type = type.args[2]) # Strip Tracker from Tracker.<...> 
  type isa QuoteNode && (type = type.value) # Unwrap a QuoteNode
  return type
end