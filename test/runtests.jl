using Test
using LinearAlgebra
using Random
using StridedTensorXD
using StridedTensorXD: StridedView, UnsafeStridedView

Random.seed!(1234)

println("Base.Threads.nthreads() =  $(Base.Threads.nthreads())")

println("Running tests single-threaded:")
StridedTensorXD.disable_threads()
include("othertests.jl")
include("blasmultests.jl")

println("Running tests multi-threaded:")
StridedTensorXD.enable_threads()
include("othertests.jl")
include("blasmultests.jl")

StridedTensorXD.enable_threaded_mul()
include("blasmultests.jl")
