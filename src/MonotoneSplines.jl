module MonotoneSplines

include("utils.jl")
include("mono_spl.jl")
include("boot.jl")

const _py_boot = PyCall.PyNULL()
const _py_sys = PyCall.PyNULL()

function __init__()
    currentdir = @__DIR__
    # py"""
    # import sys
    # sys.path.insert(0, $(currentdir))
    # from boot import train_G
    # """
    copy!(_py_sys, pyimport("sys"))
    pushfirst!(_py_sys."path", currentdir)
    copy!(_py_boot, pyimport("boot"))
end

export check_CI,
        cubic,
        logit

end # module MonotoneSplines
