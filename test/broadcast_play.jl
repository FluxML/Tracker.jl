using Tracker: Tracker, param, TrackedArray, TrackedReal, TrackedTypes, track, back!


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