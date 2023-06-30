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
