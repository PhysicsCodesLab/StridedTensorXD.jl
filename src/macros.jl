macro strided(ex)
    ex = macroexpand(__module__, ex)
    esc(_strided(ex))
end

function _strided(ex::Expr)
    if ex.head == :. && ex.args[2] isa QuoteNode # field access: wrap whole expression in maybestrided
        return Expr(:call, :(StridedTensorXD.maybestrided), ex)
    elseif ex.head == :call && ex.args[1] isa Symbol
        if ex.args[1] == :reshape
            return Expr(:call, :(StridedTensorXD.sreshape), map(_strided, ex.args[2:end])...)
        elseif ex.args[1] == :view
            return Expr(:call, :(StridedTensorXD.sview), map(_strided, ex.args[2:end])...)
        else
            return Expr(:call, ex.args[1], map(_strided, ex.args[2:end])...)
        end
    elseif (ex.head == :(=) || ex.head == :(kw)) && ex.args[1] isa Symbol
        return Expr(ex.head, ex.args[1],
                    Expr(:call, :(StridedTensorXD.maybeunstrided), _strided(ex.args[2])))
    elseif (ex.head == :(->))
        return Expr(ex.head, ex.args[1],
                    Expr(:call, :(StridedTensorXD.maybeunstrided), _strided(ex.args[2])))
    else
        return Expr(ex.head, map(_strided, ex.args)...)
    end
end
const exclusionlist = Symbol[:(:)]
_strided(ex::Symbol) =  ex in exclusionlist ? ex : Expr(:call, :(StridedTensorXD.maybestrided), ex)
_strided(ex) = ex

maybestrided(A::AbstractStridedView) = A
maybestrided(A::AbstractArray) = StridedView(A)
maybestrided(A) = A
maybeunstrided(A::StridedView) = copy(A).parent
maybeunstrided(A) = A

macro unsafe_strided(args...)
    syms = args[1:end-1]
    ex = macroexpand(__module__, args[end]) #_strided(args[end])
    all(isa(s, Symbol) for s in syms) ||
        error("The first arguments to `@unsafe_strided` must be variable names")
    ex = Expr(:let, Expr(:block, [:($s = StridedTensorXD.UnsafeStridedView($s)) for s in syms]...), ex)
    return esc(:(GC.@preserve $(syms...) $ex))
end

# macro sfor(args...)
#     syms = args[1:end-1]
#     all(isa(s, Symbol) for s in syms) || error("The first arguments to `@sfor` must be variable names that will be usek")
#     ex = macroexpand(__module__, args[end])
#     ex = _sfor(syms, ex)
# end
