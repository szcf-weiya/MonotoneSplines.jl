module MonotoneSplines

include("utils.jl")
include("conditions.jl")
include("bspl.jl")
include("mono_spl.jl")
include("boot.jl")


function __init__()
    try
        using PyCall
        global const _py_boot = PyCall.PyNULL()
        global const _py_sys = PyCall.PyNULL()    
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
    catch e
        @warn """
        PyCall is not properly installed, which is required when the PyTorch backend. 

        But you can use the Flux backend without installing PyCall.jl.
        """
    end
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
        is_sufficient_and_necessary

if Base.VERSION >= v"1.4.2"
    include("precompile.jl")
    _precompile_()
end

end # module MonotoneSplines
