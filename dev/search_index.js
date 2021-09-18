var documenterSearchIndex = {"docs":
[{"location":"index/#Index","page":"Index","title":"Index","text":"","category":"section"},{"location":"index/","page":"Index","title":"Index","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = StridedTensorXD","category":"page"},{"location":"#Abstract","page":"Home","title":"Abstract","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Julia's default array Array{T,N} uses column‐major storage. For an array A with size(A) = (d1,d2,...,dN), its element A[i1,i2,...,iN] is located in memory at positi\u0000on i_1cdot  +(i_2 1)cdot s_2 + = 1+sum_k = 1^N(i_k 1)cdot s_k, where s_1 =1 and s_k+1 =s_kcdot d_k.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Strided arrays are dense arrays whose memory layout has a fixed spacings (strides) between its elements along every dimension. The element A[i1,i2,...,i_N] of a strided array A is located at memory positi\u0000on mathrmoffset+sum_k=1^N (i_k 1)s_k for arbitrary   offset and strides s_k.","category":"page"},{"location":"","page":"Home","title":"Home","text":"StridedTensorXD is a Julia package for working more efficiently with strided arrays. The package does not make any assumptions about the strides and provides multithreaded and cache friendly implementations for mapping, reducing, broadcasting such arrays, as well as taking views, reshaping and permuting dimensions. Most of these are simply accessible by annotating a block of standard Julia code involving broadcasting and other array operations with the macro @strided.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Provided the element type of the array is compatible with BLAS, a strided array can utilize BLAS and LAPACK routines for more efficient linear algebra routines.","category":"page"},{"location":"#Multithreading","page":"Home","title":"Multithreading","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StridedTensorXD.jl uses the new @spawn threading infrastructure.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The threading is enabled by default, and use the maximum number of threads available StridedTensorXD._NTHREADS = Threads.nthreads().","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that the value given by Threads.nthreads() is the number of execution threads we set when we start Julia, which is default to be 1. It is controlled either by using the -t or --threads command line argument or by using the JULIA_NUM_THREADS environment variable. When both are specified, then -t/--threads takes precedence. For example, to start Julia with 6 threads, we use","category":"page"},{"location":"","page":"Home","title":"Home","text":"$ julia --threads 6","category":"page"},{"location":"","page":"Home","title":"Home","text":"The number of threads can be set manually by StridedTensorXD.set_num_threads(n), where n can be any integer between 1 (no threading) and Threads.nthreads().","category":"page"},{"location":"","page":"Home","title":"Home","text":"There are also convenience functions StridedTensorXD.enable_threads(), which set the number of threads to Threads.nthreads(), and StridedTensorXD.disable_threads(), which set it to 1.","category":"page"},{"location":"","page":"Home","title":"Home","text":"There is an experimental feature (disabled by default) to apply multithreading for matrix multiplication using a divide-and-conquer strategy. It can be enabled via StridedTensorXD.enable_threaded_mul(), and disabled by  Strided.disable_threaded_mul().","category":"page"},{"location":"","page":"Home","title":"Home","text":"For matrices with a LinearAlgebra.BlasFloat element type (i.e. any of Float32, Float64, ComplexF32 or ComplexF64), this is typically not necessary as BLAS is multithreaded by default.","category":"page"},{"location":"","page":"Home","title":"Home","text":"However, it can be beneficial to implement the multithreading using Julia Tasks, which then run on Julia's threads as distributed by Julia's scheduler. Hence, this feature should likely be used in combination with LinearAlgebra.BLAS.set_num_threads(1). Performance seems to be on par (within a few percent margin) with the threading strategies of OpenBLAS and MKL. However, note that the latter call also disables any multithreading used in LAPACK (e.g. eigen, svd, qr, ...) and StridedTensorXD.jl does not help with that.","category":"page"},{"location":"#Examples","page":"Home","title":"Examples","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Running Julia with a single thread","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using StridedTensorXD\n\njulia> using BenchmarkTools\n\njulia> A = randn(4000,4000);\n\njulia> B = similar(A);\n\njulia> @btime $B .= ($A .+ $A') ./ 2;\n  126.720 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= ($A .+ $A') ./ 2;\n  44.314 ms (15 allocations: 880 bytes)\n\njulia> A = randn(1000,1000);\n\njulia> B = similar(A);\n\njulia> @btime $B .= 3 .* $A';\n  1.487 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= 3 .* $A';\n  971.260 μs (13 allocations: 672 bytes)\n\njulia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);\n  16.225 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);\n  15.407 ms (56 allocations: 3.75 KiB)\n\njulia> A = randn(32,32,32,32);\n\njulia> B = similar(A);\n\njulia> @btime permutedims!($B, $A, (4,3,2,1));\n  3.047 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided permutedims!($B, $A, (4,3,2,1));\n  1.555 ms (12 allocations: 864 bytes)\n\njulia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));\n  9.577 ms (12 allocations: 32.00 MiB)\n\njulia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));\n  5.299 ms (18 allocations: 1.72 KiB)","category":"page"},{"location":"","page":"Home","title":"Home","text":"And now with julia -t 6:","category":"page"},{"location":"","page":"Home","title":"Home","text":"julia> using StridedTensorXD\n\njulia> using BenchmarkTools\n\njulia> A = randn(4000,4000);\n\njulia> B = similar(A);\n\njulia> @btime $B .= ($A .+ $A') ./ 2;\n  126.888 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= ($A .+ $A') ./ 2;\n  23.415 ms (112 allocations: 7.95 KiB)\n\njulia> A = randn(1000,1000);\n\njulia> B = similar(A);\n\njulia> @btime $B .= 3 .* $A';\n  1.385 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= 3 .* $A';\n  486.831 μs (109 allocations: 6.70 KiB)\n\njulia> @btime $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);\n  15.286 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided $B .= $A .* exp.( -2 .* $A) .+ sin.( $A .* $A);\n  4.201 ms (153 allocations: 12.95 KiB)\n\njulia> A = randn(32,32,32,32);\n\njulia> B = similar(A);\n\njulia> @btime permutedims!($B, $A, (4,3,2,1));\n  2.976 ms (0 allocations: 0 bytes)\n\njulia> @btime @strided permutedims!($B, $A, (4,3,2,1));\n  794.465 μs (103 allocations: 9.00 KiB)\n\njulia> @btime $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));\n  11.498 ms (12 allocations: 32.00 MiB)\n\njulia> @btime @strided $B .= permutedims($A, (1,2,3,4)) .+ permutedims($A, (2,3,4,1)) .+ permutedims($A, (3,4,1,2)) .+ permutedims($A, (4,1,2,3));\n  1.969 ms (110 allocations: 15.45 KiB)","category":"page"},{"location":"#Design-principles","page":"Home","title":"Design principles","text":"","category":"section"},{"location":"#StridedView","page":"Home","title":"StridedView","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StridedTensorXD.jl is centered around the type StridedView, which provides a view into a parent array of type DenseArray such that the resulting view is strided, i.e. any dimension has an associated stride, such that e.g.","category":"page"},{"location":"","page":"Home","title":"Home","text":"getindex(A, i₁, i₂, i₃, ...) = A.op(A.parent[offset + 1 + (i₁-1)*s₁ + (i₂-1)*s₂ +\n                                      (i₃-1)*s₃ + ...])","category":"page"},{"location":"","page":"Home","title":"Home","text":"with sⱼ = stride(A, iⱼ). The operation A.op can be any of identity, conj, transpose or adjoint (the latter two are equivalent to the former two if eltype(A) <: Number). Since these operations are their own inverse, they are also used in the corresponding setindex!.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This definition enables a StridedView to be lazy (i.e. returns just another StridedView over the same parent data) under application of conj, transpose, adjoint, permutedims and indexing (getindex) with SliceIndex = Union{RangeIndex,Colon}.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The strided structure can be retained under certain reshape operations, but not all of them. Any dimension can always be split into smaller dimensions, but two subsequent dimensions i and i+1 can only be joined if stride(A,i+1) == size(A,i)*stride(A,i). Instead of overloading reshape, StridedTensorXD.jl provides a separate function sreshape which returns a StridedView over the same parent data, or throws a runtime error if this is impossible.","category":"page"},{"location":"#Broadcasting-and-map(reduce)","page":"Home","title":"Broadcasting and map(reduce)","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Whenever an expression only contains StridedViews and scalars, overloaded methods for broadcasting and functions as map(!) and mapreduce are used that exploit the known strided structure in order to evaluate the result in a more efficient way, at least for sufficiently large arrays where the overhead of the extra preparatory work is negligible. In particular, this involves choosing a blocking strategy and loop order that aims to avoid cache misses. This matters in particular if some of the StridedViews involved have strides which are not monotonously increasing, e.g. if transpose, adjoint or permutedims has been applied. The fact that the permutedims also acts lazily (whereas it creates a copy of the data in Julia base) can potentially provide a further speedup.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Furthermore, these optimized methods are implemented with support for multithreading. Thus, if Threads.nthreads() > 1 and the arrays involved are sufficiently large, performance can be boosted even for plain arrays with a strictly sequential memory layout, provided that the broadcast operation is compute bound and not memory bound (i.e. the broadcast function is sufficienlty complex).","category":"page"},{"location":"#The-@strided-macro-annotation","page":"Home","title":"The @strided macro annotation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Rather than manually wrapping every array in a StridedView, there is the macro annotation @strided some_expression, which will wrap all DenseArrays appearing in some_expression in a StridedView. Note that, because StridedViews behave lazily under indexing with ranges, this acts similar to the @views macro in Julia Base, i.e. there is no need to use a view.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The macro @strided acts as a contract, i.e. the user ensures that all array manipulations in the following expressions will preserve the strided structure. Therefore, reshape and view are are replaced by sreshape and sview respectively. As mentioned above, sreshape will throw an error if the requested new shape is incompatible with preserving the strided structure. The function sview is only defined for index arguments which are ranges, Ints or Colon (:), and will thus also throw an error if indexed by anything else.","category":"page"},{"location":"#StridedView-versus-StridedArray-and-BLAS/LAPACK-compatibility","page":"Home","title":"StridedView versus StridedArray and BLAS/LAPACK compatibility","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"StridedArray is a union type to denote arrays with a strided structure in Julia Base. Because of its definition as a type union rather than an abstract type, it is impossible to have user types be recognized as StridedArray. This is rather unfortunate, since dispatching to BLAS and LAPACK routines is based on StridedArray. As a consequence, StridedView will not fall back to BLAS or LAPACK by default. Currently, only matrix multiplication is overloaded so as to fall back to BLAS (i.e. gemm!) if possible. In general, one should not attempt use e.g. matrix factorizations or other lapack operations within the @strided context. Support for this is on the TODO list. Some BLAS inspired methods (axpy!, axpby!, scalar multiplication via mul!, rmul! or lmul!) are however overloaded by relying on the optimized yet generic map! implementation.","category":"page"},{"location":"","page":"Home","title":"Home","text":"StridedViews can currently only be created with certainty from DenseArray (typically just Array in Julia Base). For Base.SubArray or Base.ReshapedArray instances, the StridedView constructor will first act on the underlying parent array, and then try to mimic the corresponding view or reshape operation using sview and sreshape. These, however, are more limited then their Base counterparts (because they need to guarantee that the result still has a strided memory layout with respect to the new dimensions), so an error can result. However, this approach can also succeed in creating StridedView wrappers around combinations of view and reshape that are not recognised as Base.StridedArray. For example, reshape(view(randn(40,40), 1:36, 1:20), (6,6,5,4)) is not a Base.StridedArrray, and indeed, it cannot statically be inferred to be strided, from only knowing the argument types provided to view and reshape. For example, the similarly looking reshape(view(randn(40,40), 1:36, 1:20), (6,3,10,4)) is not strided. The StridedView constructor will try to act on both, and yield a runtime error in the second case. Note that Base.ReinterpretArray is currently not supported.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note again that, unlike StridedArrays, StridedViews behave lazily (i.e. still produce a view on the same parent array) under permutedims and regular indexing with ranges.","category":"page"},{"location":"#UnsafeStridedView-and-@unsafe_strided","page":"Home","title":"UnsafeStridedView and @unsafe_strided","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Based on the work of UnsafeArrays.jl there is also an UnsafeStridedView, which references the parent array via a pointer, and therefore is itself a stack allocated struct (i.e. isbitstype(UnsafeStridedView{...}) is true).","category":"page"},{"location":"","page":"Home","title":"Home","text":"It behaves in all respects the same as StridedView (they are both subtypes of AbstractStridedView), except that by itself it does not keep a reference to the parent array in a way that is visible to Julia's garbage collector. It can therefore not be the return value of an operation (in particular similar(::UnsafeStridedView, ...) -> ::StridedView) and an explicit reference to the parent array needs to be kept alive. Furthermore, UnsafeStridedView wrappers can only be created of AbstractArray{T} instances with isbitstype(T).","category":"page"},{"location":"","page":"Home","title":"Home","text":"There is a corresponding @unsafe_strided macro annotation. However, in this case the arrays in the expression need to be identified explicitly as","category":"page"},{"location":"","page":"Home","title":"Home","text":"@unsafe_strided A₁ A₂ ... some_expression","category":"page"},{"location":"","page":"Home","title":"Home","text":"because this will be translated into the expression","category":"page"},{"location":"","page":"Home","title":"Home","text":"GC.@preserve A₁ A₂ ...\nlet A₁ = UnsafeStridedView(A₁), A₂ = ...\n    some_expression\nend","category":"page"},{"location":"#Planned-features-/-wish-list","page":"Home","title":"Planned features / wish list","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Support for GPUArrays with dedicated GPU kernels?","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [StridedTensorXD]","category":"page"},{"location":"#StridedTensorXD.SliceIndex","page":"Home","title":"StridedTensorXD.SliceIndex","text":"const StridedTensorXD.SliceIndex = Union{RangeIndex,Colon}\n\nSliceIndex is the union name of standard RangeIndex and Colon, where RangeIndex = Union{Int, AbstractRange{Int}, AbstractUnitRange{Int}}.\n\n\n\n\n\n","category":"type"},{"location":"#StridedTensorXD._NTHREADS","page":"Home","title":"StridedTensorXD._NTHREADS","text":"const StridedTensorXD._NTHREADS = Ref(1)\n\nThe global constant represents the number of threads in use, default to Ref(1).\n\n\n\n\n\n","category":"constant"},{"location":"#StridedTensorXD._use_threaded_mul","page":"Home","title":"StridedTensorXD._use_threaded_mul","text":"const StridedTensorXD._use_threaded_mul = Ref(false)\n\nThe global constant represents whether we use multithreading for matrix multiplicatins, default to false.\n\n\n\n\n\n","category":"constant"},{"location":"#StridedTensorXD.AbstractStridedView","page":"Home","title":"StridedTensorXD.AbstractStridedView","text":"abstract type AbstractStridedView{T,N,F<:Union{FN,FC,FA,FT}} <: AbstractArray{T,N} end\n\nAbstract type for all kinds of StridedView.\n\nType Parameters\n\nT: the type of the elements\nN: the dimension of the array\nF<:Union{FN,FC,FA,FT}: the function denotes the relation between the strided view and                           its parent\n\nThe fields that needed in any concrete subtypes:\n\nparent::A where A<:DenseArray\nsize::NTuple{N,Int}: size of the stridedview; default to be the size of parent\nstrides::NTuple{N,Int}: strides of the stridedview; default to be strides of parent\noffset::Int: default to be 0; the position of the first element of the strided view is               offset + 1.\nop::F: default to be identity\n\n\n\n\n\n","category":"type"},{"location":"#StridedTensorXD.ParentIndex","page":"Home","title":"StridedTensorXD.ParentIndex","text":"struct ParentIndex\n    i::Int\nend\n\nThe index directly into parent array. It is used to getindex from the parent of strided view directly.\n\n\n\n\n\n","category":"type"},{"location":"#StridedTensorXD.StridedView","page":"Home","title":"StridedTensorXD.StridedView","text":"struct StridedView{T,N,A<:DenseArray,F<:Union{FN,FC,FA,FT}} <: AbstractStridedView{T,N,F}\n    parent::A\n    size::NTuple{N,Int}\n    strides::NTuple{N,Int}\n    offset::Int\n    op::F\nend\n\nA concrete type of strided view.\n\n\n\n\n\n","category":"type"},{"location":"#StridedTensorXD.StridedView-Union{Tuple{Array{S, N} where N}, Tuple{F}, Tuple{N}, Tuple{S}, Tuple{Array{S, N} where N, Tuple{Vararg{Int64, N}}}, Tuple{Array{S, N} where N, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}}, Tuple{Array{S, N} where N, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}, Int64}, Tuple{Array{S, N} where N, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}, Int64, F}} where {S, N, F}","page":"Home","title":"StridedTensorXD.StridedView","text":"StridedView(parent::Array{S},\n            size::NTuple{N,Int} = size(parent),\n            strides::NTuple{N,Int} = strides(parent),\n            offset::Int = 0,\n            op::F = identity) where {S, N, F}\n\nA constructor of StridedView instance where parent can be an Array with any kind of elements S.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.StridedView-Union{Tuple{F}, Tuple{N}, Tuple{A}, Tuple{A, Tuple{Vararg{Int64, N}}}, Tuple{A, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}}, Tuple{A, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}, Int64}, Tuple{A, Tuple{Vararg{Int64, N}}, Tuple{Vararg{Int64, N}}, Int64, F}} where {A<:DenseArray, N, F}","page":"Home","title":"StridedTensorXD.StridedView","text":"StridedView(parent::A,\n            size::NTuple{N,Int} = size(parent),\n            strides::NTuple{N,Int} = strides(parent),\n            offset::Int = 0,\n            op::F = identity) where {A<:DenseArray, N, F}\n\nA constructor of StridedView instance where parent is a general DenseArray, but not in the form of Array{S}.\n\n\n\n\n\n","category":"method"},{"location":"#Base._mapreduce_dim-Tuple{Any, Any, NamedTuple{(:init,), T} where T<:Tuple, StridedTensorXD.AbstractStridedView, Colon}","page":"Home","title":"Base._mapreduce_dim","text":"\n\n\n\n","category":"method"},{"location":"#Base.adjoint-Tuple{StridedView{var\"#s47\", 2, A, F} where {var\"#s47\"<:Number, A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}","page":"Home","title":"Base.adjoint","text":"adjoint(a::StridedView{<:Number,2})\nadjoint(a::StridedView{<:Any,2})\n\nExtend LinearAlgebra.adjoint.\n\nIf the elements of a.parent are just numbers, it returns a new strided view with transposed size and strides, and also replace a.op with _conj(a.op).\n\nOtherwise, in general, it returns a new strided view with transposed size and strides and also replace a.op with _adjoint(a.op).\n\n\n\n\n\n","category":"method"},{"location":"#Base.conj-Tuple{StridedView{var\"#s47\", N, A, F} where {var\"#s47\"<:Real, N, A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}","page":"Home","title":"Base.conj","text":"conj(a::StridedView)\n\nExtend Base.conj. Leave the strided view a invariant if its elements are real. Return a new strided view which replace a.op with _conj(a.op) in general.\n\n\n\n\n\n","category":"method"},{"location":"#Base.elsize-Union{Tuple{Type{var\"#s27\"} where var\"#s27\"<:(StridedTensorXD.AbstractStridedView{T, N, F} where {N, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}})}, Tuple{T}} where T","page":"Home","title":"Base.elsize","text":"elsize(::Type{<:AbstractStridedView{T}})\n\nReturn the size of T which is the type of the elements.\n\n\n\n\n\n","category":"method"},{"location":"#Base.map-Tuple{typeof(conj), StridedView{var\"#s47\", N, A, F} where {var\"#s47\"<:Real, N, A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}","page":"Home","title":"Base.map","text":"map(::FC, a::StridedView{<:Real})\nmap(::FT, a::StridedView{<:Number})\nmap(::FA, a::StridedView{<:Number})\nmap(::FC, a::StridedView)\nmap(::FT, a::StridedView)\nmap(::FA, a::StridedView)\n\nExtend Base.map. Return a new strided view with the correct a.op marked by the type.\n\n\n\n\n\n","category":"method"},{"location":"#Base.mapreduce-Tuple{Any, Any, StridedTensorXD.AbstractStridedView}","page":"Home","title":"Base.mapreduce","text":"Base.mapreduce(f, op, A::AbstractStridedView; dims=:, kw...)\n\nApply function f to each element(s) in A, and then reduce the result using the binary function op along dimensions of A. dims is a vector specifying the dimensions to reduce, and the keyword argument kw... = init is the initial value to use in the reductions. In general, it will be necessary to provide init to work with empty collections.\n\n\n\n\n\n","category":"method"},{"location":"#Base.permutedims-Union{Tuple{N}, Tuple{StridedView{var\"#s47\", N, A, F} where {var\"#s47\", A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Any}} where N","page":"Home","title":"Base.permutedims","text":"permutedims(a::StridedView{<:Any,N}, p) where {N}\n\nExtend Base.permutedims. Return a new strided view with permuted size and strides according to permutation p.\n\n\n\n\n\n","category":"method"},{"location":"#Base.stride-Tuple{StridedView{var\"#s47\", 0, A, F} where {var\"#s47\", A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Int64}","page":"Home","title":"Base.stride","text":"Base.stride(a::StridedView{<:Any, N}, n::Int) where N\n\nReturn the distance in memory (in number of elements) between adjacent elements in dimension n.\n\n\n\n\n\n","category":"method"},{"location":"#Base.transpose-Tuple{StridedView{var\"#s47\", 2, A, F} where {var\"#s47\"<:Number, A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}","page":"Home","title":"Base.transpose","text":"transpose(a::StridedView{<:Number,2})\ntranspose(a::StridedView{<:Any,2})\n\nExtend LinearAlgebra.transpose.\n\nIf the elements of a.parent are just numbers, it returns a new strided view with transposed size and strides.\n\nOtherwise, in general, it returns a new strided view with transposed size and strides and also replace a.op with _transpose(a.op).\n\nNote that the reason for these two behaviors is similar to the difference between Base.permutedims and LinearAlgebra.transpose in the original Julia codes. In the general case, the elements of a.parent could be Arrays themselves, therefore, the mark _transpose(a.op) plays the role which is known as \"acting recursively\".\n\n\n\n\n\n","category":"method"},{"location":"#Base.unsafe_convert-Union{Tuple{T}, Tuple{Type{Ptr{T}}, StridedView{T, N, A, F} where {N, A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}} where T","page":"Home","title":"Base.unsafe_convert","text":"Return the native address of an array.\n\n\n\n\n\n","category":"method"},{"location":"#Base.view-Union{Tuple{N}, Tuple{StridedTensorXD.AbstractStridedView{var\"#s25\", N, F} where {var\"#s25\", F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Vararg{Union{Colon, Int64, AbstractRange{Int64}}, N}}} where N","page":"Home","title":"Base.view","text":"Base.view(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}\n\nReturn a new strided view from the index I of the old strided view a.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.lmul!-Tuple{Number, StridedTensorXD.AbstractStridedView}","page":"Home","title":"LinearAlgebra.lmul!","text":"LinearAlgebra.lmul!(α::Number, dst::AbstractStridedView)\n\nReplace dst by α * dst, where α is a number.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.mul!-Union{Tuple{N}, Tuple{StridedTensorXD.AbstractStridedView{var\"#s27\", N, F} where {var\"#s27\"<:Number, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, StridedTensorXD.AbstractStridedView{var\"#s25\", N, F} where {var\"#s25\"<:Number, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Number}} where N","page":"Home","title":"LinearAlgebra.mul!","text":"LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N},\n                    src::AbstractStridedView{<:Number,N}, α::Number) where {N}\n\nReturn dst = src * α, where α is a number.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.mul!-Union{Tuple{N}, Tuple{StridedTensorXD.AbstractStridedView{var\"#s33\", N, F} where {var\"#s33\"<:Number, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Number, StridedTensorXD.AbstractStridedView{var\"#s34\", N, F} where {var\"#s34\"<:Number, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}} where N","page":"Home","title":"LinearAlgebra.mul!","text":"LinearAlgebra.mul!(dst::AbstractStridedView{<:Number,N}, α::Number,\n                    src::AbstractStridedView{<:Number,N}) where {N}\n\nReturn dst = α * src, where α is a number.\n\n\n\n\n\n","category":"method"},{"location":"#LinearAlgebra.rmul!-Tuple{StridedTensorXD.AbstractStridedView, Number}","page":"Home","title":"LinearAlgebra.rmul!","text":"LinearAlgebra.rmul!(dst::AbstractStridedView, α::Number)\n\nReplace dst by dst * α, where α is a number.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.__init__-Tuple{}","page":"Home","title":"StridedTensorXD.__init__","text":"StridedTensorXD.__init__()\n\nInitialize by setting the number of threads as Threads.nthreads().\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.__mul!-Tuple{StridedTensorXD.AbstractStridedView{var\"#s25\", 2, F} where {var\"#s25\", F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, StridedTensorXD.AbstractStridedView{var\"#s44\", 2, F} where {var\"#s44\", F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, StridedTensorXD.AbstractStridedView{var\"#s45\", 2, F} where {var\"#s45\", F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Any, Any}","page":"Home","title":"StridedTensorXD.__mul!","text":"__mul!(C::AbstractStridedView{<:Any,2}, A::AbstractStridedView{<:Any,2},\n        B::AbstractStridedView{<:Any,2}, α, β)\n\nReturn C = A*B*α + C*β. This implementation is faster than LinearAlgebra.generic_matmatmul.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._computeind-Tuple{Tuple{}, Tuple{}}","page":"Home","title":"StridedTensorXD._computeind","text":"_computeind(indices::NTuple{N,Int}, strides::NTuple{N,Int}) where {N}\n\nReturn the index in a.parent that corresponds to the input index indices of the strided view. indices is the index in the Cartesian form.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._computereshapestrides-Tuple{Tuple{}, Tuple{}, Tuple{}}","page":"Home","title":"StridedTensorXD._computereshapestrides","text":"_computereshapestrides(newsize::Dims, oldsize::Dims{N}, oldstrides::Dims{N}) where {N}\n\nReturn the new strides after we reshape the strided view from the oldsize to the newsize. The input oldstrides is the old strides before reshape.\n\nUsed in sreshape for the concrete type of strided view.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._computeviewoffset-Union{Tuple{N}, Tuple{Tuple{Vararg{Int64, N}}, Tuple{Vararg{Union{Colon, Int64, AbstractRange{Int64}}, N}}}} where N","page":"Home","title":"StridedTensorXD._computeviewoffset","text":"_computeviewoffset(strides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}\n\nReturn the new offset after selecting range I from the old strided view.\n\nIf I[i] is a Colon, return 0 for that dimension; if I[i] is an Int or a RangeIndex, return first(I[1])-1)*strides[1] for that dimension. Add all offsets for each dimension to give the total new offset.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._computeviewsize-Union{Tuple{N}, Tuple{Tuple{Vararg{Int64, N}}, Tuple{Vararg{Union{Colon, Int64, AbstractRange{Int64}}, N}}}} where N","page":"Home","title":"StridedTensorXD._computeviewsize","text":"_computeviewsize(oldsize::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}\n\nReturn size after selecting the range I in the old strided view.\n\nIf I[i] is an Int, remove that dimension; if I[i] is a Colon, that dimension remain invariant; if I[i] is a RangeIndex, the new size in that dimension is equal to length(I[i]).\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._computeviewstrides-Union{Tuple{N}, Tuple{Tuple{Vararg{Int64, N}}, Tuple{Vararg{Union{Colon, Int64, AbstractRange{Int64}}, N}}}} where N","page":"Home","title":"StridedTensorXD._computeviewstrides","text":"_computeviewstrides(oldstrides::NTuple{N,Int}, I::NTuple{N,SliceIndex}) where {N}\n\nReturn the strides after selecting the range I in the old strided view.\n\nIf I[i] is an Int, remove that dimension; if I[i] is a Colon, leave that dimension invariant; if I[i] is a RangeIndex, the new stride should be oldstrides[1]*step(I[1]).\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._normalizestrides-Tuple{Tuple{}, Tuple{}}","page":"Home","title":"StridedTensorXD._normalizestrides","text":"_normalizestrides(size::Dims{N}, strides::Dims{N})\n\nReturn the nomalized strides with the following conventions: If for i, size[i] == 1, set the corresponding strides[i] as size[i-1]*strides[i-1]. If for i, size[i] == 0, set the ith and all the following strides elements after i to be size[i-1]*strides[i-1].\n\nUsed to construct general strided view type.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._simplify-Tuple{Tuple{}, Tuple{}}","page":"Home","title":"StridedTensorXD._simplify","text":"_simplify(size::Dims{N}, strides::Dims{N}) where {N}\n\nSimplify the size and strides based on the following rules: (1) If size[1] == 1, move this dimension to the last one. (2) If size[1]*strides[1] == tailstrides[1], combine the second dimension to the first and get the new first size as size[1]*tailsize[1] with strides[1]. We also add a one dimensional space to the last, with strides tailsize[1]*tailstrides[1].\n\nUsed in sreshape for the concrete type of strided view.\n\nExamples\n\njulia> _simplify((2, 1, 3, 4, 1), (1, 2, 2, 6, 24))\n((24, 1, 1, 1, 1), (1, 24, 24, 2, 24))\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD._threaded_blas_mul!-Union{Tuple{T}, Tuple{StridedTensorXD.AbstractStridedView{T, 2, F} where F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}, StridedTensorXD.AbstractStridedView{T, 2, F} where F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}, StridedTensorXD.AbstractStridedView{T, 2, F} where F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}, Any, Any, Any}} where T<:Union{Float32, Float64, ComplexF32, ComplexF64}","page":"Home","title":"StridedTensorXD._threaded_blas_mul!","text":"_threaded_blas_mul!(C::AbstractStridedView{T,2}, A::AbstractStridedView{T,2},\n    B::AbstractStridedView{T,2}, α, β, nthreads) where {T<:LinearAlgebra.BlasFloat}\n\nReturn C = A*B*α + C*β. Use BLAS function gemm! and use multiple threads if nthreads > 1 and size(C,1)*size(C,2) > 1024.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.blasstrides-Tuple{StridedView{var\"#s47\", 2, A, F} where {var\"#s47\", A<:DenseArray, F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}}","page":"Home","title":"StridedTensorXD.blasstrides","text":"blasstrides(a::StridedView{<:Any,2})\n\nIf size(a, 2) == 1 and stride(a, 1) == 1, return a new stride view with the strides (1,size(a,1)) to be compatible with the BLAS function.\n\nOtherwise, return the original a.\n\nNOTE: If size(a,2) == 1, stride(a,2) does not really have a meaning, as you are not taking any steps in the 2nd dimension. Nonetheless, BLAS still expects that stride(a,2) == size(a,1). But sometimes in StridedTensorXD.jl, if you get this a from permuting another array, it might be that stride(a,2) has some other value.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.disable_threaded_mul-Tuple{}","page":"Home","title":"StridedTensorXD.disable_threaded_mul","text":"StridedTensorXD.disable_threaded_mul()\n\nDisable the use of multithreading for matrix multiplicatins.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.disable_threads-Tuple{}","page":"Home","title":"StridedTensorXD.disable_threads","text":"StridedTensorXD.disable_threads()\n\nSet the number of threads to 1.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.enable_threaded_mul-Tuple{}","page":"Home","title":"StridedTensorXD.enable_threaded_mul","text":"StridedTensorXD.enable_threaded_mul()\n\nEnable the use of multithreading for matrix multiplicatins.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.enable_threads-Tuple{}","page":"Home","title":"StridedTensorXD.enable_threads","text":"StridedTensorXD.enable_threads()\n\nSet the number of threads to the maximum value available.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.get_num_threads-Tuple{}","page":"Home","title":"StridedTensorXD.get_num_threads","text":"StridedTensorXD.get_num_threads() -> Int\n\nGet the current number of threads in use now.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.getblasmatrix-Union{Tuple{StridedTensorXD.AbstractStridedView{T, 2, F} where F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Tuple{T}} where T<:Union{Float32, Float64, ComplexF32, ComplexF64}","page":"Home","title":"StridedTensorXD.getblasmatrix","text":"getblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}\n\nReturn the strided view with the compatible form of strides that can be used in BLAS functions and the label which determine whether to transpose an argument in BLAS functions.\n\n'N': The input matrix X is not transposed or conjugated.\n\n'T': The input matrix X will be transposed.\n\n'C': The input matrix X will be conjugated and transposed.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.isblasmatrix-Union{Tuple{StridedTensorXD.AbstractStridedView{T, 2, F} where F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Tuple{T}} where T<:Union{Float32, Float64, ComplexF32, ComplexF64}","page":"Home","title":"StridedTensorXD.isblasmatrix","text":"isblasmatrix(A::AbstractStridedView{T,2}) where {T<:LinearAlgebra.BlasFloat}\n\nCheck whether the strided view A is a blas matrix.\n\nThe definition of the blas matrix is that:\n\nCase 1: if A.op == identity, it requires either stride(A,1) == 1 or stride(A,2) == 1;\n\nCase 2: if A.op == conj, it requires stride(A, 2) == 1.\n\nCase 3: If A.op is neither identity nor conj, A is not a blas matrix.\n\nThe definition of blas matrix comes from the facts that the BLAS module requires contiguous matrix columns.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.set_num_threads-Tuple{Int64}","page":"Home","title":"StridedTensorXD.set_num_threads","text":"StridedTensorXD.set_num_threads(n::Int)\n\nSet the number of threads, i.e., set the value of the global constant _NTHREADS.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.sreshape","page":"Home","title":"StridedTensorXD.sreshape","text":"sreshape(a::AbstractStridedView,newsize::Vararg{Int})\n\nReturn a strided view of that has the same parent with a, but with newsize.\n\nThis function is needed to be implemented for each concrete type of strided view.\n\n\n\n\n\n","category":"function"},{"location":"#StridedTensorXD.sview-Union{Tuple{N}, Tuple{StridedTensorXD.AbstractStridedView{var\"#s25\", N, F} where {var\"#s25\", F<:Union{typeof(adjoint), typeof(conj), typeof(identity), typeof(transpose)}}, Vararg{Union{Colon, Int64, AbstractRange{Int64}}, N}}} where N","page":"Home","title":"StridedTensorXD.sview","text":"sview(a::AbstractStridedView{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}\nsview(a::AbstractStridedView, I::SliceIndex)\nsview(a::DenseArray{<:Any,N}, I::Vararg{SliceIndex,N}) where {N}\nsview(a::DenseArray, I::SliceIndex)\n\nReturn a view from index I of the strided view or a DenseArrayas a new strided view.\n\n\n\n\n\n","category":"method"},{"location":"#StridedTensorXD.use_threaded_mul-Tuple{}","page":"Home","title":"StridedTensorXD.use_threaded_mul","text":"StridedTensorXD.use_threaded_mul()\n\nCheck whether we are using multithreading for matrix multiplicatins.\n\n\n\n\n\n","category":"method"}]
}
