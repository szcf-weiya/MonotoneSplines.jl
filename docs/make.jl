ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
using Documenter
using MonotoneSplines
using Literate
indir = joinpath(@__DIR__, "..", "examples")
outdir = joinpath(@__DIR__, "src", "examples")
files = ["monofit.jl", "monofit_mlp.jl"]
for file in files
    Literate.markdown(joinpath(indir, file), outdir; credit = false)
end

makedocs(sitename="MonotoneSplines.jl",
        pages = [
            "Home" => "index.md",
            "Examples" => [
                "Monotone Fitting" => "examples/monofit.md",
                "MLP Generator" => "examples/monofit_mlp.md"
            ],
            "API" => "api.md"
        ]
)