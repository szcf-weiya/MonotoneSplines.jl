# This section illustrates the space of γ for monotonicity with a toy example J = 4.
using LaTeXStrings
using Plots

## illustration of space of γ for monotonicity
function plot_intervals(; step = 0.1, γ3 = 3, γ4 = 4, boundary = false)
    f(γ1, γ2) = γ3 ≥ γ2 ≥ γ1
    function xstar(γ1, γ2, γ3 = 3, γ4 = 4)
        w = 1 - (γ4 - 2γ3 + γ2) / (γ4 - 3γ3 + 3γ2 - γ1)
        return 1.0 * (w > 1) + w * (0 < w < 1)
    end
    b2(x) = (1-x)^2
    b3(x) = 2x * (1-x)
    b4(x) = x^2
    function fp(γ1, γ2, γ3 = 3, γ4 = 4)
        t = xstar(γ1, γ2)
        return 3(b2(t) * (γ2 - γ1) + b3(t) * (γ3 - γ2) + b4(t) * (γ4 - γ3))
    end

    xs = range(-15, 15, step = step)
    ys = range(-15, 15, step = step)
    z = [f(xi, yi) for yi in ys, xi in xs] ## TODO: compare with for for
    ## heatmap(z) #cons: overlap

    z2 = [(abs(xstar(xi, yi, γ3, γ4) - 0.5) >= 0.5) * (yi ≥ xi) for yi in ys, xi in xs]
    ## heatmap!(z2) #cons: overlap
    z3 = [(fp(xi, yi, γ3, γ4) ≥ 0) * (yi ≥ xi) for yi in ys, xi in xs]

    cidx = findall(z .> 0)
    i1 = [i[1] for i in cidx]
    i2 = [i[2] for i in cidx]
    yt = [-10, -5, 0, 3, 5, 10]
    plt = scatter(xs[i2], ys[i1], 
            markershape = :vline,  # more clear
            ## markershape = :x, # slightly dense 
            markersize = 3, xlim = (-10, 10), ylim = (-10, 10), 
            xlab = latexstring("\$\\gamma_1\$"), ylab = latexstring("\$\\gamma_2\$"),
            title = latexstring("\$\\gamma_3 = $γ3, \\gamma_4 = $γ4\$"),
            yticks = (yt, string.(yt)),
            label = "sufficient", legend = :bottomright)
    cidx3 = findall(max.(z2, z3) .> 0)
    i31 = [i[1] for i in cidx3]
    i32 = [i[2] for i in cidx3]
    scatter!(plt, xs[i32], ys[i31], 
            markershape = :hline, 
            markersize = 3, alpha = 0.5, label = "sufficient & necessary")
    plot!(plt, xs, ys, label = "necessary", fillrange = 10, fillalpha = 0.3, linealpha = 0)
    ## calculated boundary
    γ2s = range(γ3, 10, step = 0.01)
    γ1s = γ2s .- (γ2s .- γ3).^2 / (γ4 - γ3)
    if boundary
        plot!(plt, γ1s, γ2s, label = "")
    end
    return plt
    ## plot_intervals(γ3 = 3, γ4 = 3.1, step = 0.4)
    ## savefig("~/PGitHub/overleaf/MonotoneFitting/res/conditions_case1.pdf")
    ## plot_intervals(γ3 = 3, γ4 = 5, step = 0.4)
    ## savefig("~/PGitHub/overleaf/MonotoneFitting/res/conditions_case2.pdf")
    ## savefig("~/PGitHub/overleaf/MonotoneFitting/res/conditions_case2_boundary.pdf")
end

# reproduce the figure in the paper
plot_intervals()