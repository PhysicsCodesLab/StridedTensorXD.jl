```@meta
CurrentModule = StridedTensorXD
```
# Abstract
Julia's default array `Array{T,N}` uses column‐major storage. For an array `A` with
`size(A) = (d1,d2,...,dN)`, its element `A[i1,i2,...,iN]` is located in memory at
positi on ``i_1\cdot —— +(i_2 −1)\cdot s_2 +... = 1+\sum_{k = 1}^N(i_k −1)\cdot s_k``, where
``s_1 =1`` and ``s_{k+1} =s_k\cdot d_k``.

Strided arrays are dense arrays whose memory layout has a fixed spacings (strides) between
its elements along every dimension. The element `A[i1,i2,...,i_N]` of a strided array `A`
is located at memory positi on ``\mathrm{offset}+\sum_{k=1}^N (i_k −1)⋅s_k`` for arbitrary  
`offset` and strides ``s_k``.

[StridedTensorXD](https://github.com/PhysicsCodesLab/StridedTensorXD.jl) is a Julia package for working
more efficiently with strided arrays. The package does not make any assumptions about the
strides and provides multithreaded and cache friendly implementations for mapping,
reducing, broadcasting such arrays, as well as taking views, reshaping and permuting
dimensions. Most of these are simply accessible by annotating a block of standard Julia
code involving broadcasting and other array operations with the macro `@strided`.

Provided the element type of the array is compatible with BLAS, a strided array can utilize
BLAS and LAPACK routines for more efficient linear algebra routines.

# Multithreading

`StridedTensorXD.jl` uses the new `@spawn` threading infrastructure.

The threading is enabled by default, and use the maximum number of threads available
`StridedTensorXD._NTHREADS = Threads.nthreads()`.

Note that the value given by `Threads.nthreads()` is the number of execution threads
we set when we start Julia, which is default to be `1`. It is controlled either by using the
`-t` or `--threads` command line argument or by using the `JULIA_NUM_THREADS` environment
variable. When both are specified, then `-t`/`--threads` takes precedence.
For example, to start Julia with 6 threads, we use
```bash
$ julia --threads 6
```

The number of threads can be set manually by `StridedTensorXD.set_num_threads(n)`, where `n` can
be any integer between `1` (no threading) and `Threads.nthreads()`.

There are also convenience functions `StridedTensorXD.enable_threads()`, which set the number of
threads to `Threads.nthreads()`, and `StridedTensorXD.disable_threads()`, which set it to `1`.

There is an experimental feature (disabled by default) to apply multithreading
for matrix multiplication using a divide-and-conquer strategy. It can be enabled via
`StridedTensorXD.enable_threaded_mul()`, and disabled by  `Strided.disable_threaded_mul()`.

For matrices with a `LinearAlgebra.BlasFloat` element type (i.e. any of `Float32`,
`Float64`, `ComplexF32` or `ComplexF64`), this is typically not necessary as BLAS is
multithreaded by default.

However, it can be beneficial to implement the multithreading using Julia Tasks, which then
run on Julia's threads as distributed by Julia's scheduler. Hence, this feature should
likely be used in combination with `LinearAlgebra.BLAS.set_num_threads(1)`. Performance
seems to be on par (within a few percent margin) with the threading strategies of OpenBLAS
and MKL. However, note that the latter call also disables any multithreading used in LAPACK
(e.g. `eigen`, `svd`, `qr`, ...) and StridedTensorXD.jl does not help with that.

# Examples

Running Julia with a single thread

```julia
julia> using StridedTensorXD

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  126.720 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  44.314 ms (15 allocations: 880 bytes)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  1.487 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  971.260 μs (13 allocations: 672 bytes)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  16.225 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  15.407 ms (56 allocations: 3.75 KiB)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  3.047 ms (0 allocations: 0 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  1.555 ms (12 allocations: 864 bytes)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  9.577 ms (12 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  5.299 ms (18 allocations: 1.72 KiB)
```

And now with `julia -t 6`:
```julia
julia> using StridedTensorXD

julia> using BenchmarkTools

julia> A = randn(4000,4000);

julia> B = similar(A);

julia> @btime $B .= ($A .+ $A') ./ 2;
  126.888 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= ($A .+ $A') ./ 2;
  23.415 ms (112 allocations: 7.95 KiB)

julia> A = randn(1000,1000);

julia> B = similar(A);

julia> @btime $B .= 3 .* $A';
  1.385 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= 3 .* $A';
  486.831 μs (109 allocations: 6.70 KiB)

julia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  15.286 ms (0 allocations: 0 bytes)

julia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);
  4.201 ms (153 allocations: 12.95 KiB)

julia> A = randn(32,32,32,32);

julia> B = similar(A);

julia> @btime permutedims!($B, $A, (4,3,2,1));
  2.976 ms (0 allocations: 0 bytes)

julia> @btime @strided permutedims!($B, $A, (4,3,2,1));
  794.465 μs (103 allocations: 9.00 KiB)

julia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  11.498 ms (12 allocations: 32.00 MiB)

julia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));
  1.969 ms (110 allocations: 15.45 KiB)
```

# Design principles

## `StridedView`

StridedTensorXD.jl is centered around the type `StridedView`, which provides a view into a parent
array of type `DenseArray` such that the resulting view is strided, i.e. any dimension
has an associated stride, such that e.g.
```julia
getindex(A, i₁, i₂, i₃, ...) = A.op(A.parent[offset + 1 + (i₁-1)*s₁ + (i₂-1)*s₂ +
                                      (i₃-1)*s₃ + ...])
```
with `sⱼ = stride(A, iⱼ)`. The operation `A.op` can be any of `identity`, `conj`,
`transpose` or `adjoint` (the latter two are equivalent to the former two if `eltype(A) <:
Number`). Since these operations are their own inverse, they are also used in the
corresponding `setindex!`.

This definition enables a `StridedView` to be lazy (i.e. returns just another `StridedView`
over the same parent data) under application of `conj`, `transpose`, `adjoint`,
`permutedims` and indexing (`getindex`) with `SliceIndex = Union{RangeIndex,Colon}`.

The strided structure can be retained under certain `reshape` operations, but
not all of them. Any dimension can always be split into smaller dimensions, but two
subsequent dimensions `i` and `i+1` can only be joined if `stride(A,i+1) ==
size(A,i)*stride(A,i)`. Instead of overloading `reshape`, StridedTensorXD.jl provides a separate
function `sreshape` which returns a `StridedView` over the same parent data, or throws a
runtime error if this is impossible.

## Broadcasting and `map(reduce)`

Whenever an expression only contains `StridedView`s and scalars,
overloaded methods for broadcasting and functions as `map(!)` and `mapreduce` are used that
exploit the known strided structure in order to evaluate the result in a more efficient way,
at least for sufficiently large arrays where the overhead of the extra preparatory work is
negligible. In particular, this involves choosing a blocking strategy and loop order that
aims to avoid cache misses. This matters in particular if some of the `StridedView`s
involved have strides which are not monotonously increasing, e.g. if `transpose`, `adjoint`
or `permutedims` has been applied. The fact that the `permutedims` also acts lazily (whereas it
creates a copy of the data in Julia base) can potentially provide a further speedup.

Furthermore, these optimized methods are implemented with support for multithreading. Thus,
if `Threads.nthreads() > 1` and the arrays involved are sufficiently large, performance can
be boosted even for plain arrays with a strictly sequential memory layout, provided that the
broadcast operation is compute bound and not memory bound (i.e. the broadcast function is
sufficienlty complex).

## The `@strided` macro annotation
Rather than manually wrapping every array in a `StridedView`, there is the macro annotation
`@strided some_expression`, which will wrap all `DenseArray`s appearing in `some_expression`
in a `StridedView`. Note that, because `StridedView`s behave lazily under indexing with
ranges, this acts similar to the `@views` macro in Julia Base, i.e. there is no need to use
a view.

The macro `@strided` acts as a contract, i.e. the user ensures that all array manipulations
in the following expressions will preserve the strided structure. Therefore, `reshape` and
`view` are are replaced by `sreshape` and `sview` respectively. As mentioned above,
`sreshape` will throw an error if the requested new shape is incompatible with preserving
the strided structure. The function `sview` is only defined for index arguments which are
ranges, `Int`s or `Colon` (`:`), and will thus also throw an error if indexed by anything
else.

## `StridedView` versus `StridedArray` and BLAS/LAPACK compatibility

`StridedArray` is a union type to denote arrays with a strided structure in Julia Base.
Because of its definition as a type union rather than an abstract type, it is impossible to
have user types be recognized as `StridedArray`. This is rather unfortunate, since
dispatching to BLAS and LAPACK routines is based on `StridedArray`. As a consequence,
`StridedView` will not fall back to BLAS or LAPACK by default. Currently, only matrix
multiplication is overloaded so as to fall back to BLAS (i.e. `gemm!`) if possible. In
general, one should not attempt use e.g. matrix factorizations or other lapack operations
within the `@strided` context. Support for this is on the TODO list. Some BLAS inspired
methods (`axpy!`, `axpby!`, scalar multiplication via `mul!`, `rmul!` or `lmul!`) are
however overloaded by relying on the optimized yet generic `map!` implementation.

`StridedView`s can currently only be created with certainty from `DenseArray` (typically
just `Array` in Julia Base). For `Base.SubArray` or `Base.ReshapedArray` instances, the
`StridedView` constructor will first act on the underlying parent array, and then try to
mimic the corresponding view or reshape operation using `sview` and `sreshape`. These,
however, are more limited then their Base counterparts (because they need to guarantee that
the result still has a strided memory layout with respect to the new dimensions), so an
error can result. However, this approach can also succeed in creating `StridedView` wrappers
around combinations of `view` and `reshape` that are not recognised as `Base.StridedArray`.
For example, `reshape(view(randn(40,40), 1:36, 1:20), (6,6,5,4))` is not a
`Base.StridedArrray`, and indeed, it cannot statically be inferred to be strided, from only
knowing the argument types provided to `view` and `reshape`. For example, the similarly
looking `reshape(view(randn(40,40), 1:36, 1:20), (6,3,10,4))` is not strided. The
`StridedView` constructor will try to act on both, and yield a runtime error in the second
case. Note that `Base.ReinterpretArray` is currently not supported.

Note again that, unlike `StridedArray`s, `StridedView`s behave lazily (i.e. still produce a
view on the same parent array) under `permutedims` and regular indexing with ranges.

## `UnsafeStridedView` and `@unsafe_strided`
Based on the work of [UnsafeArrays.jl](https://github.com/oschulz/UnsafeArrays.jl) there is
also an `UnsafeStridedView`, which references the parent array via a pointer, and therefore
is itself a stack allocated `struct` (i.e. `isbitstype(UnsafeStridedView{...})` is true).

It behaves in all respects the same as `StridedView` (they are both subtypes of
`AbstractStridedView`), except that by itself it does not keep a reference to the parent
array in a way that is visible to Julia's garbage collector. It can therefore not be the
return value of an operation (in particular
`similar(::UnsafeStridedView, ...) -> ::StridedView`) and an explicit reference to the
parent array needs to be kept alive. Furthermore, `UnsafeStridedView` wrappers can only be
created of `AbstractArray{T}` instances with `isbitstype(T)`.

There is a corresponding `@unsafe_strided` macro annotation. However, in this case the
arrays in the expression need to be identified explicitly as
```julia
@unsafe_strided A₁ A₂ ... some_expression
```

because this will be translated into the expression
```julia
GC.@preserve A₁ A₂ ...
let A₁ = UnsafeStridedView(A₁), A₂ = ...
    some_expression
end
```

# Planned features / wish list

*   Support for `GPUArray`s with dedicated GPU kernels?

```@index
```

```@autodocs
Modules = [StridedTensorXD]
```
