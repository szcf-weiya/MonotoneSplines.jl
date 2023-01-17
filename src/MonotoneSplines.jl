module MonotoneSplines

include("utils.jl")
include("mono_spl.jl")
include("boot.jl")

const _py_boot = PyCall.PyNULL()
const _py_sys = PyCall.PyNULL()

function __init__()
    currentdir = @__DIR__
    copy!(_py_sys, pyimport("sys"))
    pushfirst!(_py_sys."path", currentdir)
    try
        copy!(_py_boot, pyimport("boot"))
    catch e
        @warn """
        PyTorch backend for MLP generator is not ready due to 
        =============
        $e
        =============
        - If you want to enable the PyTorch backend, please refer to https://github.com/szcf-weiya/MonotoneSplines.jl/blob/394a005b675ef425c8c34212b58a5c2c7a5e3f17/.github/workflows/ci.yml#L50-L56 for the installation instruction.
        - If you want to continue to use the package, it is still fine. For the MLP generator, you can choose the Flux backend, and you can use other functions in the package, such as monotone fitting with optimization toolbox.
        """
    end
end

export check_CI,
       check_acc,
        cubic,
        logit,
        logit5,
        sinhalfpi,
        mono_cs,
        mono_ss,
        mono_ss_mlp,
        ci_mono_ss,
        ci_mono_ss_mlp,
        jaccard_index,
        rcopy,
        predict

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end # module MonotoneSplines
