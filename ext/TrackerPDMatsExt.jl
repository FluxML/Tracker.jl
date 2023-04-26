module TrackerPDMatsExt

if !isdefined(Base, :get_extension)
    using ..Tracker, ..PDMats
else
    using Tracker, PDMats
end

Base.:\(A::PDMat, B::Tracker.TrackedVecOrMat) = Tracker.track(\, A, B)
Base.:\(A::PDiagMat, B::Tracker.TrackedVecOrMat) = Tracker.track(\, A, B)
Base.:\(A::ScalMat, B::Tracker.TrackedVecOrMat) = Tracker.track(\, A, B)

end
