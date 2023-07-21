module MonotoneSplines

include("utils.jl")
include("conditions.jl")
include("bspl.jl")
include("mono_spl.jl")
include("boot.jl")

function __init_pytorch__()
    @eval Main begin
        import PyCall
        export PyCall
    end
    _post_imports()
    _runtime_init()
end

_post_imports() = @eval begin
    const PyCall = Main.PyCall
end

_runtime_init() = @eval begin
    _py_boot = PyCall.PyNULL() # no need constant (it throws redefinition)
    _py_sys = PyCall.PyNULL()
    currentdir = @__DIR__
    copy!(_py_sys, PyCall.pyimport("sys"))
    pushfirst!(_py_sys."path", currentdir)
    copy!(_py_boot, PyCall.pyimport("boot"))    
end

function __init__()
    nothing
end

export check_CI,
       check_acc,
        cubic,
        logit,
        logit5,
        sinhalfpi,
        smooth_spline,
        mono_cs,
        mono_ss,
        mono_ss_mlp,
        cv_mono_ss,
        ci_mono_ss,
        ci_mono_ss_mlp,
        jaccard_index,
        rcopy,
        predict,
        bs3_τi,
        bs4_τi,
        is_sufficient,
        is_necessary,
        is_sufficient_and_necessary,
        __init_pytorch__

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end # module MonotoneSplines
