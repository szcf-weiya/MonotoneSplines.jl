function bs3_τi(τ::AbstractArray, i::Int)
    # j = i-2, i-1
    j = i-2
    r2 = (τ[j+3] - τ[j+2]) / (τ[j+3] - τ[j+1])
    j = i-1
    r1 = (τ[j+1] - τ[j]) / (τ[j+2] - τ[j])
    return [r2, r1]
end

function bs4_τi(τ::AbstractArray, i::Int)
    # evaluate at j = i-1, i-2, i-3
    hi = τ[i+1] - τ[i]
    hi_1 = τ[i] - τ[i-1]
    hi_2_3 = τ[i-2+3] - τ[i-2]
    hi_2_2 = τ[i-2+2] - τ[i-2]
    hi_1_2 = τ[i-1+2] - τ[i-1]
    hi_1_3 = τ[i-1+3] - τ[i-1]
    hi_x_2 = τ[i+2] - τ[i]

    return [hi * hi / (hi_1_2 * hi_2_3),
            hi_2_2 * hi / (hi_2_3 * hi_1_2) + hi_x_2 * hi_1 / (hi_1_3 * hi_1_2),
            hi_1 * hi_1 / (hi_1_3 * hi_1_2)]
end
