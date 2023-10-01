using Tracker: Tracker, param, TrackedArray, TrackedReal, TrackedTypes, track, back!
using LinearAlgebra: diagm, dot, LowerTriangular, norm, det, logdet, logabsdet, I, Diagonal

##
ta = param([1, 2, 3])
tb = param([2, 4, 6])
ta .+ tb
tb = ta.^3
tb = sum(sin.(1 .+ ta.^2))
back!(tb)
ta.tracker.grad


u = 2
v = 3
@enter u^v

ta = param([1 2; 3 4])
tb = logabsdet(ta)[1]
back!(tb)
back1 = tb.tracker.f.func
y = back1(1)
y[1].backing
tc = sin.(tb)
td = sum(tc)
@run back!(td)
f = (x) -> logabsdet(x)[1]
f(ta)