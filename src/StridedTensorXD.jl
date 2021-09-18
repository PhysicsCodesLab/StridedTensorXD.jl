module StridedTensorXD

import Base: parent, size, strides, tail, setindex
using Base: @propagate_inbounds, RangeIndex, Dims
using LinearAlgebra
using TupleLabXD
using TupleLabXD: StaticLength

export StridedView, @strided, @unsafe_strided, sreshape, sview

"""
    const StridedTensorXD.SliceIndex = Union{RangeIndex,Colon}

SliceIndex is the union name of standard `RangeIndex` and `Colon`, where
`RangeIndex = Union{Int, AbstractRange{Int}, AbstractUnitRange{Int}}`.
"""
const SliceIndex = Union{RangeIndex,Colon}

"""
    const StridedTensorXD._NTHREADS = Ref(1)

The global constant represents the number of threads in use, default to `Ref(1)`.
"""
const _NTHREADS = Ref(1)

"""
    StridedTensorXD.set_num_threads(n::Int)

Set the number of threads, i.e., set the value of the global constant `_NTHREADS`.
"""
function set_num_threads(n::Int)
    N = Threads.nthreads()
    if n > N
        n = N
        _set_num_threads_warn(n)
    end
    # Ref objects are dereferenced (loaded or stored) with [],
    # i.e., explicitly give a value for a bitstype reference.
    _NTHREADS[] = n
end

@noinline function _set_num_threads_warn(n)
    @warn "Maximal number of threads limited by number of Julia threads,
            setting number of threads equal to Threads.nthreads() = $n"
end

"""
    StridedTensorXD.get_num_threads() -> Int

Get the current number of threads in use now.
"""
get_num_threads() = _NTHREADS[]

"""
    StridedTensorXD.enable_threads()

Set the number of threads to the maximum value available.
"""
enable_threads() = set_num_threads(Threads.nthreads())

"""
    StridedTensorXD.disable_threads()

Set the number of threads to 1.
"""
disable_threads() = set_num_threads(1)

"""
    StridedTensorXD.__init__()

Initialize by setting the number of threads as `Threads.nthreads()`.
"""
function __init__()
    set_num_threads(Threads.nthreads())
end

"""
    const StridedTensorXD._use_threaded_mul = Ref(false)

The global constant represents whether we use multithreading for matrix multiplicatins,
default to `false`.
"""
const _use_threaded_mul = Ref(false)

"""
    StridedTensorXD.use_threaded_mul()

Check whether we are using multithreading for matrix multiplicatins.
"""
use_threaded_mul() = _use_threaded_mul[]

"""
    StridedTensorXD.enable_threaded_mul()

Enable the use of multithreading for matrix multiplicatins.
"""
function enable_threaded_mul()
    _use_threaded_mul[] = true
    return
end

"""
    StridedTensorXD.disable_threaded_mul()

Disable the use of multithreading for matrix multiplicatins.
"""
function disable_threaded_mul()
    _use_threaded_mul[] = false
    return
end

include("abstractstridedview.jl")
include("stridedview.jl")
include("unsafestridedview.jl")
include("mapreduce.jl")
include("broadcast.jl")
include("macros.jl")
end
