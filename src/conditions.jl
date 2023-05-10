using RCall

function is_sufficient(γ::AbstractArray)
    return all(diff(γ) .≥ 0)
end

function is_necessary(γ::AbstractArray, τ::AbstractArray, bbasis)
    @assert length(γ) + 4 == length(τ)
    J = length(γ)
    x = τ[4:J+1]
    dB = rcopy(R"fda::eval.basis($x, $bbasis, Lfdobj = 1)")    
    d = dB * γ
    return all(d .≥ 0)
end

function is_sufficient_and_necessary(γ::AbstractArray, τ::AbstractArray, bbasis)
    @assert length(γ) + 4 == length(τ)
    J = length(γ)
    K = J - 4
    A = zeros(J)
    for i = 0:K+1
        A[i+3] = 1/(τ[i+5] - τ[i+3]) * ((γ[i+3] - γ[i+2]) / (τ[i+6] - τ[i+3]) - (γ[i+2] - γ[i+1]) / (τ[i+5] - τ[i+2]) )
    end
    πs = zeros(K+1)
    for i = 0:K
        t = A[i+4] / (A[i+4] - A[i+3])
        πs[i+1] = t * (0 < t < 1) + 1.0 * (t ≥ 1)
    end
    ξstar = τ[4:J] .* πs + τ[5:J+1] .* (1 .- πs)
    x = vcat(τ[4:J+1], ξstar)
    dB = rcopy(R"fda::eval.basis($x, $bbasis, Lfdobj = 1)")
    d = dB * γ
    return all(d .≥ 0)
end

function test_conditions()
    ## J = 4
    γ = [1, 2, 3, 4]
    ξ = [0, 1]
    τ = vcat(zeros(3), ξ, ones(3))
    bbasis = R"fda::create.bspline.basis(breaks = $ξ, norder = 4)"
    is_sufficient(γ)
    is_necessary(γ, τ, bbasis)
    is_sufficient_and_necessary(γ, τ, bbasis)
    γ1s = range(-10, 10, step = 1)
    γ2s = range(-10, 10, step = 1)
    res = zeros(length(γ1s), length(γ2s), 3)
    for (i, γ1) in enumerate(γ1s)
        for (j, γ2) in enumerate(γ2s)
            γ = [γ1, γ2, 3, 4]
            res[i, j, 1] = is_sufficient(γ)
            res[i, j, 2] = is_necessary(γ, τ, bbasis)
            res[i, j, 3] = is_sufficient_and_necessary(γ, τ, bbasis)
        end
    end
    return sum(res, dims = (1, 2))[:]
end

## illustration of space of γ for monotonicity
function plot_intervals(; step = 0.1, γ3 = 3, γ4 = 4)
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
    # heatmap(z) #cons: overlap

    z2 = [(abs(xstar(xi, yi, γ3, γ4) - 0.5) >= 0.5) * (yi ≥ xi) for yi in ys, xi in xs]
    # heatmap!(z2) #cons: overlap
    z3 = [(fp(xi, yi, γ3, γ4) ≥ 0) * (yi ≥ xi) for yi in ys, xi in xs]

    cidx = findall(z .> 0)
    i1 = [i[1] for i in cidx]
    i2 = [i[2] for i in cidx]
    yt = [-10, -5, 0, 3, 5, 10]
    plt = scatter(xs[i2], ys[i1], 
            markershape = :vline,  # more clear
            # markershape = :x, # slightly dense 
            markersize = 3, xlim = (-10, 10), ylim = (-10, 10), 
            xlab = latexstring("\$\\gamma_1\$"), ylab = latexstring("\$\\gamma_2\$"),
            # title = "γ3 = $γ3, γ4 = $γ4",
            title = latexstring("\$\\gamma_3 = $γ3, \\gamma_4 = $γ4\$"),
            yticks = (yt, string.(yt)),
            label = "sufficient", legend = :bottomright)
    cidx3 = findall(max.(z2, z3) .> 0)
    i31 = [i[1] for i in cidx3]
    i32 = [i[2] for i in cidx3]
    scatter!(plt, xs[i32], ys[i31], 
            markershape = :hline, 
            # markershape = :+, 
            markersize = 3, alpha = 0.5, label = "sufficient & necessary")

    #Plots.abline!(plt, 1, 0, label = "necessary", fill = (10, 0.3, :auto), linealpha = 0)
    #Plots.abline!(plt, 1, 0, label = "necessary", fillrange = 10, fillalpha = 0.3, linealpha = 0)
    plot!(plt, xs, ys, label = "necessary", fillrange = 10, fillalpha = 0.3, linealpha = 0)
    return plt
    # plot_intervals(γ3 = 3, γ4 = 3.1, step = 0.4)
    # savefig("~/PGitHub/overleaf/MonotoneFitting/res/conditions_case1.pdf")
    # plot_intervals(γ3 = 3, γ4 = 5, step = 0.4)
    # savefig("~/PGitHub/overleaf/MonotoneFitting/res/conditions_case2.pdf")
end

