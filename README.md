# MonotoneSplines.jl

[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://szcf-weiya.github.io/MonotoneSplines.jl/dev)
[![CI](https://github.com/szcf-weiya/MonotoneSplines.jl/actions/workflows/ci.yml/badge.svg)](https://github.com/szcf-weiya/MonotoneSplines.jl/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/szcf-weiya/MonotoneSplines.jl/branch/master/graph/badge.svg?token=bAtqskenbG)](https://codecov.io/gh/szcf-weiya/MonotoneSplines.jl)

*MonotoneSplines.jl* is a Julia package for monotone splines, which impose a monotonicity constraint on the smoothing splines. 

Check the following paper for more details.

> Lijun Wang, Xiaodan Fan, Huabai Li, and Jun S. Liu (2024).  “Monotone Cubic B-Splines with a Neural-Network Generator.” Journal of Computational and Graphical Statistics. https://doi.org/10.1080/10618600.2024.2431070

```
@article{wangMonotoneCubicBSplines2023b,
  title = {Monotone Cubic {{B}}-Splines with a Neural-Network Generator},
  author = {Wang, Lijun and Fan, Xiaodan and Li, Huabai and Liu, Jun S.},
  year= {2024},
  journal = {Journal of Computational and Graphical Statistics},
  date = {2024-09-28},
  eprint = {2307.01748},
  eprinttype = {arxiv},
  eprintclass = {astro-ph, stat},
  doi = {10.1080/10618600.2024.2431070},
  url = {https://doi.org/10.1080/10618600.2024.2431070},
  urldate = {2023-09-21},
  pages = {in press},
  keywords = {published}
}
```

## :hammer_and_wrench: Installation

:bangbang::bangbang: **R is required, but Python is optional.** :bangbang::bangbang:

The package offers two deep-learning backends for the Multi-Layer Perceptrons (MLP) generator:

- [Flux](https://github.com/FluxML/Flux.jl): pure Julia
- PyTorch: Call Python via [PyCall.jl](https://github.com/JuliaPy/PyCall.jl)

The statistical software R has offered several powerful packages on splines, such as `splines` and `fda`. We do not reinvent the wheel. Instead, we stand on the shoulders of the R giant by calling several basic core functions with the help of [RCall.jl](https://github.com/JuliaInterop/RCall.jl).

### :gear: use system R and Python 

*MonotoneSplines.jl* is available at the General Registry, so you can easily install the package in the Julia session after typing `]`,

```julia
julia> ]
(@v1.8) pkg> add MonotoneSplines
```

By default, both `PyCall.jl` and `RCall.jl` would try to use the system Python and R, respectively (more details can be found in their repos). 

### :ladder: standalone R and Python via Conda.jl

Another easy way is to install standalone `R` and `Python` via [Conda.jl](https://github.com/JuliaPy/Conda.jl) (no need to install `Conda.jl` explicitly) by specifying the following environmental variables before adding the package.

```julia
julia> ENV["PYTHON"]=""
julia> ENV["R_HOME"]="*"
julia> ]
(@v1.8) pkg> add MonotoneSplines
```

If you use the standalone R provided by Conda in Julia, the dependent R packages will be automatically installed during the building step.

## :books: Documentation

The documentation <https://hohoweiya.xyz/MonotoneSplines.jl/stable/> elaborates on the usage of the package via various simulation examples and an interesting astrophysics application.
