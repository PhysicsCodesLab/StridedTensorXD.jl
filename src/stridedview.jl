# StridedView
"""
    struct StridedView{T,N,A<:DenseArray,F<:Union{FN,FC,FA,FT}} <: AbstractStridedView{T,N,F}
        parent::A
        size::NTuple{N,Int}
        strides::NTuple{N,Int}
        offset::Int
        op::F
    end

A concrete type of strided view.
"""
struct StridedView{T,N,A<:DenseArray,F<:Union{FN,FC,FA,FT}} <: AbstractStridedView{T,N,F}
    parent::A
    size::NTuple{N,Int}
    strides::NTuple{N,Int}
    offset::Int
    op::F
end

"""
    StridedView(parent::Array{S},
                size::NTuple{N,Int} = size(parent),
                strides::NTuple{N,Int} = strides(parent),
                offset::Int = 0,
                op::F = identity) where {S, N, F}

A constructor of StridedView instance where `parent` can be an `Array` with any kind of
elements `S`.
"""
function StridedView(parent::Array{S},
                        size::NTuple{N,Int} = size(parent),
                        strides::NTuple{N,Int} = strides(parent),
                        offset::Int = 0,
                        op::F = identity) where {S, N, F}

    T = Base.promote_op(op, S)
    # reshape array to vector in order to reduce number of element types
    StridedView{T,N,Vector{S},F}(reshape(parent, length(parent)), size,
                                    _normalizestrides(size, strides), offset, op)
end

"""
    StridedView(parent::A,
                size::NTuple{N,Int} = size(parent),
                strides::NTuple{N,Int} = strides(parent),
                offset::Int = 0,
                op::F = identity) where {A<:DenseArray, N, F}

A constructor of StridedView instance where `parent` is a general `DenseArray`, but not in
the form of `Array{S}`.
"""
function StridedView(parent::A,
                        size::NTuple{N,Int} = size(parent),
                        strides::NTuple{N,Int} = strides(parent),
                        offset::Int = 0,
                        op::F = identity) where {A<:DenseArray, N, F}

    T = Base.promote_op(op, eltype(parent))
    StridedView{T,N,A,F}(parent, size, strides, offset, op)
end

StridedView(a::StridedView) = a
StridedView(a::Adjoint) = StridedView(a')'
StridedView(a::Transpose) = transpose(StridedView(transpose(a)))
StridedView(a::Base.SubArray) = sview(StridedView(a.parent), a.indices...)
StridedView(a::Base.ReshapedArray) = sreshape(StridedView(a.parent), a.dims)

Base.parent(a::StridedView) = a.parent
Base.size(a::StridedView) = a.size
Base.strides(a::StridedView) = a.strides
offset(a::StridedView) = a.offset

"""
    Base.stride(a::StridedView{<:Any, N}, n::Int) where N

Return the distance in memory (in number of elements) between adjacent elements in
dimension `n`.
"""
Base.stride(a::StridedView{<:Any, 0}, n::Int) = 1
Base.stride(a::StridedView{<:Any, N}, n::Int) where N =
    (n <= N) ? a.strides[n] : a.strides[N]*a.size[N]

Base.IndexStyle(::Type{<:StridedView}) = Base.IndexCartesian()

Base.similar(a::StridedView, ::Type{T}, dims::NTuple{N,Int}) where {N,T} =
    StridedView(similar(a.parent, T, dims))

Base.copy(a::StridedView) = copy!(similar(a), a)

"Return the native address of an array."
Base.unsafe_convert(::Type{Ptr{T}}, a::StridedView{T}) where T =
    pointer(a.parent, a.offset+1)

Base.dataids(a::StridedView) = Base.dataids(a.parent)

@inline function sreshape(a::StridedView, newsize::Dims)
    if any(isequal(0), newsize)
        any(isequal(0), size(a)) || throw(DimensionMismatch())
        newstrides = one.(newsize)
    else
        newstrides = _computereshapestrides(newsize, _simplify(size(a), strides(a))...)
    end
    StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end

"""
    conj(a::StridedView)

Extend `Base.conj`.
Leave the strided view `a` invariant if its elements are real.
Return a new strided view which replace `a.op` with `_conj(a.op)` in general.
"""
Base.conj(a::StridedView{<:Real}) = a
Base.conj(a::StridedView) = StridedView(a.parent, a.size, a.strides, a.offset, _conj(a.op))

"""
    permutedims(a::StridedView{<:Any,N}, p) where {N}

Extend `Base.permutedims`.
Return a new strided view with permuted size and strides according to permutation `p`.
"""
@inline function Base.permutedims(a::StridedView{<:Any,N}, p) where {N}
    (length(p) == N && TupleLabXD.isperm(p)) ||
        throw(ArgumentError("Invalid permutation of length $N: $p"))
    newsize = TupleLabXD._permute(a.size, p)
    newstrides = TupleLabXD._permute(a.strides, p)
    return StridedView(a.parent, newsize, newstrides, a.offset, a.op)
end

"""
    transpose(a::StridedView{<:Number,2})
    transpose(a::StridedView{<:Any,2})

Extend `LinearAlgebra.transpose`.

If the elements of `a.parent` are just numbers, it returns a new strided view with
transposed size and strides.

Otherwise, in general, it returns a new strided view with transposed size and strides
and also replace `a.op` with `_transpose(a.op)`.

Note that the reason for these two behaviors is similar to the difference between
`Base.permutedims` and `LinearAlgebra.transpose` in the original Julia codes.
In the general case, the elements of `a.parent` could be `Array`s themselves, therefore,
the mark `_transpose(a.op)` plays the role which is known as "acting recursively".
"""
LinearAlgebra.transpose(a::StridedView{<:Number,2}) = permutedims(a, (2,1))
LinearAlgebra.transpose(a::StridedView{<:Any,2}) = # act recursively, like Base
    permutedims(StridedView(a.parent, a.size, a.strides, a.offset, _transpose(a.op)), (2,1))

"""
    adjoint(a::StridedView{<:Number,2})
    adjoint(a::StridedView{<:Any,2})

Extend `LinearAlgebra.adjoint`.

If the elements of `a.parent` are just numbers, it returns a new strided view with
transposed size and strides, and also replace `a.op` with `_conj(a.op)`.

Otherwise, in general, it returns a new strided view with transposed size and strides
and also replace `a.op` with `_adjoint(a.op)`.
"""
LinearAlgebra.adjoint(a::StridedView{<:Number,2}) = permutedims(conj(a), (2,1))
LinearAlgebra.adjoint(a::StridedView{<:Any,2}) = # act recursively, like Base
    permutedims(StridedView(a.parent, a.size, a.strides, a.offset, _adjoint(a.op)), (2,1))

"""
    map(::FC, a::StridedView{<:Real})
    map(::FT, a::StridedView{<:Number})
    map(::FA, a::StridedView{<:Number})
    map(::FC, a::StridedView)
    map(::FT, a::StridedView)
    map(::FA, a::StridedView)

Extend `Base.map`. Return a new strided view with the correct `a.op` marked by the type.
"""
Base.map(::FC, a::StridedView{<:Real}) = a
Base.map(::FT, a::StridedView{<:Number}) = a
Base.map(::FA, a::StridedView{<:Number}) = conj(a)
Base.map(::FC, a::StridedView) =
    StridedView(a.parent, a.size, a.strides, a.offset, _conj(a.op))
Base.map(::FT, a::StridedView) =
    StridedView(a.parent, a.size, a.strides, a.offset, _transpose(a.op))
Base.map(::FA, a::StridedView) =
    StridedView(a.parent, a.size, a.strides, a.offset, _adjoint(a.op))

@inline function Base.getindex(a::StridedView{<:Any,N}, I::Vararg{Int,N}) where N
    @boundscheck checkbounds(a, I...)
    @inbounds r = a.op(a.parent[a.offset+_computeind(I, a.strides)])
    return r
end
@inline function Base.setindex!(a::StridedView{<:Any,N}, v, I::Vararg{Int,N}) where N
    @boundscheck checkbounds(a, I...)
    @inbounds a.parent[a.offset+_computeind(I, a.strides)] = a.op(v)
    return a
end

@inline function Base.getindex(a::StridedView{<:Any,N},
        I::Vararg{SliceIndex,N}) where N
    StridedView(a.parent, _computeviewsize(a.size, I), _computeviewstrides(a.strides, I),
                a.offset + _computeviewoffset(a.strides, I), a.op)
end

@propagate_inbounds Base.getindex(a::StridedView, I::ParentIndex) =
    a.op(getindex(a.parent, I.i))
@propagate_inbounds Base.setindex!(a::StridedView, v, I::ParentIndex) =
    (setindex!(a.parent, a.op(v), I.i); return a)

"""
    blasstrides(a::StridedView{<:Any,2})

If `size(a, 2) == 1` and `stride(a, 1) == 1`, return a new stride view with the strides
`(1,size(a,1))` to be compatible with the BLAS function.

Otherwise, return the original `a`.

NOTE: If `size(a,2) == 1`, `stride(a,2)` does not really have a meaning, as you are not
taking any steps in the 2nd dimension. Nonetheless, BLAS still expects that
`stride(a,2) == size(a,1)`. But sometimes in `StridedTensorXD.jl`, if you get this a from
permuting another array, it might be that `stride(a,2)` has some other value.
"""
function blasstrides(a::StridedView{<:Any,2})
    # canonialize strides to make compatible with gemm
    if size(a, 2) == 1 && stride(a, 1) == 1
        return StridedView(a.parent, a.size, (1, size(a,1)), a.offset, a.op)
    else
        return a
    end
end
