# This section illustrates the space of γ for monotonicity with a toy example J = 4.
using LaTeXStrings
using Plots
using RCall
using MonotoneSplines

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

# we can also evaluate the space volumes under different conditions using Monte Carlo methods
function cmpr_volume(a = -10, b = 10, step = 1)
    ξ = [0, 1]
    τ = vcat(zeros(3), ξ, ones(3))
    bbasis = R"fda::create.bspline.basis(breaks = $ξ, norder = 4)"
    num_suff_nece = 0
    num_suff = 0
    for γ1 in a:step:b
        for γ2 in a:step:b
            for γ3 in a:step:b
                for γ4 in a:step:b
                    γ = [γ1, γ2, γ3, γ4]
                    num_suff_nece += is_sufficient_and_necessary(γ, τ, bbasis)
                    num_suff += is_sufficient(γ)
                end
            end
        end
    end
    r = num_suff / num_suff_nece
    println("num_suff = $num_suff, num_suff_nece = $num_suff_nece, suff/suff_nece = $r")
    return num_suff, num_suff_nece
end

cmpr_volume(-10, 10, 2)

# with smaller grid size, the ratio can be smaller, but it takes a longer time to evaluate.


# plot the curves that do not satisfy the condition
function plot_curves()
    ξ = [0, 1]
    τ = vcat(zeros(3), ξ, ones(3))
    bbasis = R"fda::create.bspline.basis(breaks = $ξ, norder = 4)"
    x = 0:0.05:1.0
    B = rcopy(R"fda::eval.basis($x, $bbasis)")
    y7 = B * [-2, 7, 3, 5]
    y5 = B * [-2, 5, 3, 5]
    mfit7 = mono_cs(x, y7, 7)
    mfit5 = mono_cs(x, y5, 5)
    plot(x, B * [-2, 3, 3, 5], label = latexstring("\$J=4, \\gamma_2 = 3\$"), legend = :bottomright, xlab = L"x", ylab = L"y")
    plot!(x, B * [-2, 5, 3, 5], label = latexstring("\$J=4,\\gamma_2 = 5\$"))
    plot!(x, B * [-2, 7, 3, 5], label = latexstring("\$J=4,\\gamma_2 = 7\$"))
    plot!(x, mfit5.fitted, label = "J = 5", ls = :dash, lw = 2)
    plot!(x, mfit7.fitted, label = "J = 7", ls = :dash, lw = 2)
end

plot_curves()