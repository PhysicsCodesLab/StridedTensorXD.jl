const FN = typeof(identity) # `typeof(identity)` gives an object which is a DataType
const FC = typeof(conj)
const FA = typeof(adjoint)
const FT = typeof(transpose)
_conj(::FN) = conj
_conj(::FC) = identity
_conj(::FA) = transpose
_conj(::FT) = adjoint
_transpose(::FN) = transpose
_transpose(::FC) = adjoint
_transpose(::FA) = conj
_transpose(::FT) = identity
_adjoint(::FN) = adjoint
_adjoint(::FC) = transpose
_adjoint(::FA) = identity
_adjoint(::FT) = conj

"""
    abstract type AbstractStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N} end

Abstract type for all kinds of StridedView.

# Type Parameters
- `T`: the type of the elements
- `N`: the dimension of the array
- `F<:Union{FN,FC,FA,FT}`: the function denotes the relation between the strided view and
                            its parent

The fields that needed in any concrete subtypes:
- `parent::A` where `A<:DenseArray`
- `size::NTuple{N,Int}`: size of the stridedview; default to be the size of `parent`
- `strides::NTuple{N,Int}`: strides of the stridedview; default to be strides of `parent`
- `offset::Int`: default to be 0; the position of the first element of the strided view is
                `offset + 1`.
- `op::F`: default to be `identity`
"""
abstract type AbstractStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N} end

"""
    elsize(::Type{<:AbstractStridedView{T}})

Return the size of `T` which is the type of the elements.
"""
Base.elsize(::Type{<:AbstractStridedView{T}}) where {T} =
    Base.isbitstype(T) ? sizeof(T) :
        (Base.isbitsunion(T) ? Base.bitsunionsize(T) : sizeof(Ptr))

# Methods based on map!
Base.copy!(dst::AbstractStridedView{<:Any,N}, src::AbstractStridedView{<:Any,N}) where {N} =
    map!(identity, dst, src)
Base.conj!(a::AbstractStridedView{<:Real}) = a
Base.conj!(a::AbstractStridedView) = map!(conj, a, a)
LinearAlgebra.adjoint!(dst::AbstractStridedView{<:Any,N},
    src::AbstractStridedView{<:Any,N}) where {N} = copy!(dst, adjoint(src))
Base.permutedims!(dst::AbstractStridedView{<:Any,N},
    src::AbstractStridedView{<:Any,N}, p) where {N} = copy!(dst, permutedims(src, p))

"""
    _normalizestrides(size::Dims{N}, strides::Dims{N})

Return the nomalized strides with the following conventions:
If for `i`, `size[i] == 1`, set the corresponding `strides[i]` as `size[i-1]*strides[i-1]`.
If for `i`, `size[i] == 0`, set the `i`th and all the following strides elements after `i`
to be `size[i-1]*strides[i-1]`.

Used to construct general strided view type.
"""
_normalizestrides(size::Tuple{}, strides::Tuple{}) = strides
function _normalizestrides(size::Dims{N}, strides::Dims{N}) where {N}
    if size[1] == 1
        strides = Base.setindex(strides, 1, 1)
    elseif size[1] == 0
        strides = one.(strides)
    end
    for i = 2:N
        if size[i] == 1
            strides = Base.setindex(strides, size[i-1]*strides[i-1], i)
        elseif size[i] == 0
            for j = i:N
                strides = Base.setindex(strides, size[i-1]*strides[i-1], j)
            end
        end
    end
    return strides
end

"""
    _simplify(size::Dims{N}, strides::Dims{N}) where {N}

Simplify the `size` and `strides` based on the following rules:
(1) If `size[1] == 1`, move this dimension to the last one.
(2) If `size[1]*strides[1] == tailstrides[1]`, combine the second dimension to the first
and get the new first size as `size[1]*tailsize[1]` with `strides[1]`. We also add a one
dimensional space to the last, with strides `tailsize[1]*tailstrides[1]`.

Used in `sreshape` for the concrete type of strided view.

# Examples
```julia-repl
julia> _simplify((2, 1, 3, 4, 1), (1, 2, 2, 6, 24))
((24, 1, 1, 1, 1), (1, 24, 24, 2, 24))
```
"""
_simplify(size::Tuple{}, strides::Tuple{}) = size, strides
_simplify(size::Dims{1}, strides::Dims{1}) = size, strides
function _simplify(size::Dims{N}, strides::Dims{N}) where {N}
    tailsize, tailstrides = _simplify(tail(size), tail(strides))
    if size[1] == 1
        return (tailsize..., 1), (tailstrides..., strides[1])
    elseif size[1]*strides[1] == tailstrides[1]
        return (size[1]*tailsize[1], tail(tailsize)..., 1),
            (strides[1], tail(tailstrides)..., tailsize[1]*tailstrides[1])
    else
        return (size[1], tailsize...), (strides[1], tailstrides...)
    end
end

"""
    _computereshapestrides(newsize::Dims, oldsize::Dims{N}, oldstrides::Dims{N}) where {N}

Return the new strides after we reshape the strided view from the `oldsize` to the
`newsize`. The input `oldstrides` is the old strides before reshape.

Used in `sreshape` for the concrete type of strided view.
"""
_computereshapestrides(newsize::Tuple{}, oldsize::Tuple{}, oldstrides::Tuple{}) = oldstrides
function _computereshapestrides(newsize::Tuple{}, oldsize::Dims{N}, oldstrides::Dims{N}) where {N}
    all(isequal(1), oldsize) || throw(DimensionMismatch())
    return ()
end
function _computereshapestrides(newsize::Dims, oldsize::Tuple{}, oldstrides::Tuple{})
    all(isequal(1), newsize) || throw(DimensionMismatch())
    return newsize
end
function _computereshapestrides(newsize::Dims, oldsize::Dims{N}, oldstrides::Dims{N}) where {N}
    d,r = divrem(oldsize[1], newsize[1]) # The quotient and remainder (x÷y, x%y)
    # `oldsize[1]` should always be an integer multiple of `newsize[1]`.
    if r == 0
        s1 = oldstrides[1]
        if d == 1
            oldsize = (tail(oldsize)..., 1)
            oldstrides = (tail(oldstrides)..., newsize[1]*s1)
            return (s1, _computereshapestrides(tail(newsize), oldsize, oldstrides)...)
        else
            oldsize = (d, tail(oldsize)...)
            oldstrides = (newsize[1]*s1, tail(oldstrides)...)
            return (s1, _computereshapestrides(tail(newsize), oldsize, oldstrides)...)
        end
    else
        if prod(newsize) != prod(oldsize)
            throw(DimensionMismatch())
        else
            throw(ReshapeException())
        end
    end
end

"""
    sreshape(a::AbstractStridedView,newsize::Vararg{Int})

Return a strided view of that has the same parent with `a`, but with `newsize`.

This function is needed to be implemented for each concrete type of strided view.
"""
function sreshape end
sreshape(a::AbstractStridedView, newsize::Vararg{Int}) = sreshape(a, newsize)

struct ReshapeException <: Exception end
Base.show(io::IO, e::ReshapeException) = print(io, "Cannot produce a reshaped StridedView
    without allocating, try sreshape(copy(array), newsize) or fall back to reshape(array,
    newsize)")

# Used in getindex and setindex of the strided view
"""
    _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}

Return the index in `a.parent` that corresponds to the input index `indices` of the strided
view. `indices` is the index in the Cartesian form.
"""
@inline _computeind(indices::Tuple{}, strides::Tuple{}) = 1
@inline _computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N} =
    (indices[1]-1)*strides[1] + _computeind(tail(indices), tail(strides))

"""
    _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}

Return size after selecting the range `I` in the old strided view.

If `I[i]` is an Int, remove that dimension; if `I[i]` is a `Colon`, that dimension remain
invariant; if `I[i]` is a `RangeIndex`, the new size in that dimension is equal to
`length(I[i])`.
"""
@inline function _computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewsize(tail(oldsize), tail(I))
    elseif isa(I[1], Colon)
        return (oldsize[1], _computeviewsize(tail(oldsize), tail(I))...)
    else
        return (length(I[1]), _computeviewsize(tail(oldsize), tail(I))...)
    end
end
_computeviewsize(::Tuple{}, ::Tuple{}) = ()

"""
    _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}

Return the strides after selecting the range `I` in the old strided view.

If `I[i]` is an Int, remove that dimension; if `I[i]` is a `Colon`, leave that dimension
invariant; if `I[i]` is a `RangeIndex`, the new stride should be `oldstrides[1]*step(I[1])`.
"""
@inline function _computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Int)
        return _computeviewstrides(tail(oldstrides), tail(I))
    elseif isa(I[1], Colon)
        return (oldstrides[1], _computeviewstrides(tail(oldstrides), tail(I))...)
    else
        return (oldstrides[1]*step(I[1]), _computeviewstrides(tail(oldstrides), tail(I))...)
    end
end
_computeviewstrides(::Tuple{}, ::Tuple{}) = ()

"""
    _computeviewoffset(strides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}

Return the new offset after selecting range `I` from the old strided view.

If `I[i]` is a Colon, return `0` for that dimension; if `I[i]` is an Int or a RangeIndex,
return `first(I[1])-1)*strides[1]` for that dimension. Add all offsets for each dimension
to give the total new offset.
"""
@inline function _computeviewoffset(strides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}
    if isa(I[1], Colon)
        return _computeviewoffset(tail(strides), tail(I))
    else
        return (first(I[1])-1)*strides[1]+_computeviewoffset(tail(strides), tail(I))
    end
end
_computeviewoffset(::Tuple{}, ::Tuple{}) = 0

"""
    struct ParentIndex
        i::Int
    end

The index directly into parent array. It is used to `getindex` from the `parent` of strided
view directly.
"""
struct ParentIndex
    i::Int
end

"""
    sview(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}
    sview(a::AbstractStridedView, I::SliceIndex)
    sview(a::DenseArray{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}
    sview(a::DenseArray, I::SliceIndex)

Return a view from index `I` of the strided view or a `DenseArray`as a new strided view.
"""
@inline sview(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(a, I...)
@inline sview(a::AbstractStridedView, I::SliceIndex) =
    getindex(sreshape(a, (length(a),)), I...)
@inline sview(a::DenseArray{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(StridedView(a), I...)
@inline sview(a::DenseArray, I::SliceIndex) =
    getindex(sreshape(StridedView(a), (length(a),)), I...)

"""
    Base.view(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}

Return a new strided view from the index `I` of the old strided view `a`.
"""
@inline Base.view(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N} =
    getindex(a, I...)

# linear algebra
"""
    LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, α::Number,
                        src::AbstractStridedView{<:Number,N}) where {N}

Return `dst = α * src`, where `α` is a number.
"""
function LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, α::Number,
                            src::AbstractStridedView{<:Number,N}) where {N}
    if α == 1
        copy!(dst, src)
    else
        dst .= α .* src
    end
    return dst
end

"""
    LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N},
                        src::AbstractStridedView{<:Number,N}, α::Number) where {N}

Return `dst = src * α`, where `α` is a number.
"""
function LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N},
                            src::AbstractStridedView{<:Number,N}, α::Number) where {N}
    if α == 1
        copy!(dst, src)
    else
        dst .= src .* α
    end
    return dst
end

"""
    LinearAlgebra.rmul!(dst::AbstractStridedView, α::Number)

Replace `dst` by `dst * α`, where `α` is a number.
"""
LinearAlgebra.rmul!(dst::AbstractStridedView, α::Number) = mul!(dst, dst, α)

"""
    LinearAlgebra.lmul!(α::Number, dst::AbstractStridedView)

Replace `dst` by `α * dst`, where `α` is a number.
"""
LinearAlgebra.lmul!(α::Number, dst::AbstractStridedView) = mul!(dst, α, dst)

"""
    isblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}

Check whether the strided view `A` is a blas matrix.

The definition of the blas matrix is that:

Case 1: if `A.op == identity`, it requires either `stride(A,1) == 1` or `stride(A,2) == 1`;

Case 2: if `A.op == conj`, it requires `stride(A, 2) == 1`.

Case 3: If `A.op` is neither `identity` nor `conj`, `A` is not a blas matrix.

The definition of blas matrix comes from the facts that the BLAS module requires contiguous
matrix columns.
"""
function isblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        return stride(A,1) == 1 || stride(A,2) == 1
    elseif A.op == conj
        return stride(A, 2) == 1
    else # should never happen
        return false
    end
end

"""
    getblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}

Return the strided view with the compatible form of strides that can be used in BLAS
functions and the label which determine whether to transpose an argument in BLAS functions.

'N': The input matrix X is not transposed or conjugated.

'T': The input matrix X will be transposed.

'C': The input matrix X will be conjugated and transposed.
"""
function getblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}
    if A.op == identity
        if stride(A,1) == 1
            return blasstrides(A), 'N'
        else
            return blasstrides(transpose(A)), 'T'
        end
    else
        return blasstrides(adjoint(A)), 'C'
    end
end

"""
    _threaded_blas_mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2},
        B::AbstractStridedView{T,2}, α, β, nthreads) where {T<:LinearAlgebra.BlasFloat}

Return `C = A*B*α + C*β`. Use BLAS function `gemm!` and use multiple threads if
`nthreads > 1` and `size(C,1)*size(C,2) > 1024`.
"""
function _threaded_blas_mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2},
        B::AbstractStridedView{T,2}, α, β, nthreads) where {T<:LinearAlgebra.BlasFloat}
    m, n = size(C)
    m == size(A, 1) && n == size(B, 2) || throw(DimensionMismatch())
    if nthreads == 1 || m*n < 1024
        A2, CA = getblasmatrix(A)
        B2, CB = getblasmatrix(B)
        C2 = blasstrides(C)
        LinearAlgebra.BLAS.gemm!(CA, CB, convert(T, α), A2, B2, convert(T, β), C2)
    else
        if m > n
            m2 = round(Int, m/16)*8
            nthreads2 = nthreads >> 1
            t = Threads.@spawn _threaded_blas_mul!(C[1:m2, :], A[1:m2, :], B, α, β, nthreads2)
            _threaded_blas_mul!(C[m2+1:m, :], A[m2+1:m, :], B, α, β, nthreads - nthreads2)
            wait(t)
            return C
        else
            n2 = round(Int, n/16)*8
            nthreads2 = nthreads >> 1
            t = Threads.@spawn _threaded_blas_mul!(C[:, 1:n2], A, B[:, 1:n2], α, β, nthreads2)
            _threaded_blas_mul!(C[:, n2+1:n], A, B[:, n2+1:n], α, β, nthreads - nthreads2)
            wait(t)
            return C
        end
    end
end

"""
    __mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2},
            B::AbstractStridedView{<:Any,2}, α, β)

Return `C = A*B*α + C*β`.
This implementation is faster than LinearAlgebra.generic_matmatmul.
"""
function __mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2},
                B::AbstractStridedView{<:Any,2}, α, β)
    (size(C,1) == size(A,1) && size(C,2) == size(B,2) && size(A,2) == size(B,1)) ||
        throw(DimensionMatch("A has size $(size(A)), B has size $(size(B)),
                                C has size $(size(C))"))
    m,n = size(C)
    k = size(A,2)
    A2 = sreshape(A, (m, 1, k))
    B2 = sreshape(permutedims(B,(2,1)), (1, n, k))
    C2 = sreshape(C, (m, n, 1))

    if α == 0 || k == 0
        rmul!(C, β)
    elseif α == 1
        if β == 0
            _mapreducedim!(*, +, zero, (m,n,k), (C2,A2,B2))
        elseif β == 1
            _mapreducedim!(*, +, nothing, (m,n,k), (C2,A2,B2))
        else
            _mapreducedim!(*, +, x->x*β, (m,n,k), (C2,A2,B2))
        end
    else
        f = (x,y)->(x*y*α)
        if β == 0
            _mapreducedim!(f, +, zero, (m,n,k), (C2,A2,B2))
        elseif β == 1
            _mapreducedim!(f, +, nothing, (m,n,k), (C2,A2,B2))
        else
            _mapreducedim!(f, +, x->x*β, (m,n,k), (C2,A2,B2))
        end
    end
    return C
end

# here we will have C.op == :identity && stride(C,1) < stride(C,2)
function _mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2},
                B::AbstractStridedView{T,2}, α, β) where {T<:LinearAlgebra.BlasFloat}
    if stride(C,1) == 1 && isblasmatrix(A) && isblasmatrix(B)
        nthreads = use_threaded_mul() ? get_num_threads() : 1
        _threaded_blas_mul!(C, A, B, α, β, nthreads)
    else
        return __mul!(C, A, B, α, β)
    end
end

function LinearAlgebra.mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{<:Any,2},
    B::AbstractStridedView{<:Any,2}, α::Number = true, β::Number = false) where {T}
    if !(eltype(C) <: LinearAlgebra.BlasFloat && eltype(A) == eltype(B) == eltype(C))
        return __mul!(C, A, B, α, β)
    end
    # C.op is identity or conj
    if C.op == conj
        if stride(C,1) < stride(C,2)
            _mul!(conj(C), conj(A), conj(B), conj(α), conj(β))
        else
            _mul!(C', B', A', conj(α), conj(β))
        end
    elseif stride(C,1) > stride(C,2)
        _mul!(transpose(C), transpose(B), transpose(A), α, β)
    else
        _mul!(C, A, B, α, β)
    end
    return C
end

function LinearAlgebra.axpy!(a::Number, X::AbstractStridedView{<:Number,N},
                                Y::AbstractStridedView{<:Number,N}) where {N}
    if a == 1
        Y .= X .+ Y
    else
        Y .= a .* X .+ Y
    end
    return Y
end
function LinearAlgebra.axpby!(a::Number, X::AbstractStridedView{<:Number,N}, b::Number,
                                Y::AbstractStridedView{<:Number,N}) where {N}
    if b == 1
        axpy!(a, X, Y)
    elseif b == 0
        mul!(Y, a, X)
    else
        Y .= a .* X .+ b .* Y
    end
    return Y
end

# Converting back to other DenseArray type:
function Base.convert(T::Type{<:DenseArray}, a::AbstractStridedView)
    b = T(undef, size(a))
    copy!(StridedView(b), a)
    return b
end
# following method because of ambiguity warning
function Base.convert(::Type{T}, a::AbstractStridedView) where {T<:Array}
    b = T(undef, size(a))
    copy!(StridedView(b), a)
    return b
end
function Base.convert(::Type{Array}, a::AbstractStridedView{T}) where {T}
    b = Array{T}(undef, size(a))
    copy!(StridedView(b), a)
    return b
end
