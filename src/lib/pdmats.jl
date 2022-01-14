using .PDMats

Base.:\(A::PDMat, B::TrackedVecOrMat) = Tracker.track(\, A, B)
Base.:\(A::PDiagMat, B::TrackedVecOrMat) = Tracker.track(\, A, B)
Base.:\(A::ScalMat, B::TrackedVecOrMat) = Tracker.track(\, A, B)
