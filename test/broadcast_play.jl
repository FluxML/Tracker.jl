using Tracker: Tracker, param, TrackedArray, TrackedReal, TrackedTypes, track, back!
using LinearAlgebra: diagm, dot, LowerTriangular, norm, det, logdet, logabsdet, I, Diagonal
using ChainRules: ChainRules, rrule, NoTangent

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

## stripp off the NoTangent()
y, back = rrule(sin, 1)
bk = back(1)
ids = findall(e->e != NoTangent(), bk)
bk[ids]

##
A = rand(3, 4)
x = param(rand(4))
y = A*x
z = sum(y)
back!(z)

##
spatial_rank = 2
x = rand(repeat([10], spatial_rank)..., 3, 2)
w = rand(repeat([3], spatial_rank)..., 3, 4)
cdims = DenseConvDims(x, w)
xt = param(x)
wt = param(w)
zt = conv(xt, wt, cdims)
st = sum(zt)
@run back!(st)