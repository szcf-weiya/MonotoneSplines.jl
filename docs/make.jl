ENV["PLOTS_TEST"] = "true"
ENV["GKSwstype"] = "100"
using Documenter
using MonotoneSplines
using Literate
indir = joinpath(@__DIR__, "..", "examples")
outdir = joinpath(@__DIR__, "src", "examples")
files = ["monofit.jl", "ph.jl", 
        "monofit_mlp.jl", "monoci_mlp.jl", 
        "monofit_mlp_beta_loss.jl",
        "diff_sort.jl", "conditions.jl"]
for file in files
    Literate.markdown(joinpath(indir, file), outdir; credit = false)
end
# copy data file
cp(joinpath(indir, "ph.dat"), joinpath(outdir, "ph.dat"), force = true)

makedocs(sitename="MonotoneSplines.jl",
        pages = [
            "Home" => "index.md",
            "Examples" => [
                "Monotone Fitting" => "examples/monofit.md",
                "Conditions" => "examples/conditions.md",
                "Application: Polarization-hole" => "examples/ph.md",
                "MLP Generator (fitting curve)" => "examples/monofit_mlp.md",
                "MLP Generator (confidence band)" => "examples/monoci_mlp.md",
                "MLP Generator (beta loss)" => "examples/monofit_mlp_beta_loss.md",
                "Differentiable Sort" => "examples/diff_sort.md"
            ],
            "API" => "api.md"
        ]        
)

deploydocs(
    repo = "github.com/szcf-weiya/MonotoneSplines.jl"
)