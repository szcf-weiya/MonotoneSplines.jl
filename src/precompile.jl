function _precompile_()
    ccall(:jl_generating_output, Cint, ()) == 1 || return nothing
    Base.precompile(Tuple{Core.kwftype(typeof(mono_ss_mlp)),NamedTuple{(:λl, :λu, :device, :prop_nknots), Tuple{Float64, Float64, Symbol, Float64}},typeof(mono_ss_mlp),Vector{Float64},Vector{Float64}})   # time: 6.314397
    #isdefined(MonotoneSplines, Symbol("#losses#26")) && Base.precompile(Tuple{getfield(MonotoneSplines, Symbol("#losses#26")),Vector{Float64}})   # time: 0.13651001
    Base.precompile(Tuple{Core.kwftype(typeof(train_G)),NamedTuple{(:λl, :λu, :device), Tuple{Float64, Float64, Symbol}},typeof(train_G),Vector{Float64},Matrix{Float64},Matrix{Float64}})   # time: 0.07528695
end