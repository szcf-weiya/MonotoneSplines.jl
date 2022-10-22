```@meta
EditURL = "<unknown>/../examples/monofit.jl"
```

````@example monofit
using MonotoneSplines
using Plots
using Random
using RCall
````

## Cubic Curve

````@example monofit
n = 100
seed = 1234
σ = 0.2
Random.seed!(seed)
x = rand(n) * 2 .- 1
y = x .^3 + randn(n) * σ
λs = exp.(range(-10, -4, length = 100));
nothing #hide
````

Perform cross-validation for monotone fitting with smoothing splines,

````@example monofit
@time errs, B, L, J = MonotoneSplines.cv_mono_ss(x, y, λs)
````

Then plot the CV curve

````@example monofit
scatter(log.(λs), errs, title = "seed = $seed")
````

Then we can choose `λ` which minimized the CV error.

````@example monofit
idx = argmin(errs)
λopt = λs[idx]
````

Fit with `λopt`

````@example monofit
βhat = MonotoneSplines.mono_ss(B, y, L, J, λopt);
nothing #hide
````

Alternatively,

````@example monofit
βhat, yhat = MonotoneSplines.mono_ss(x, y, λopt);
nothing #hide
````

Plot it

````@example monofit
scatter(x, y)
scatter!(x, yhat)
````

We can also compare it with `smooth.spline`,

````@example monofit
spl = R"smooth.spline($x, $y)"
````

it also determine `λ` by cross-validation,

````@example monofit
λ = rcopy(R"$spl$lambda")
````

we can plot its fitting values together,

````@example monofit
yhat_ss = rcopy(R"predict($spl, $x)$y")
scatter!(x, yhat_ss)
````

For ease of demonstrating other examples, we wrap up the above procedures as functions

````@example monofit
function demo_mono_ss(x, y, λs)
    errs, B, L, J = MonotoneSplines.cv_mono_ss(x, y, λs)
    fig1 = plot(log.(λs), errs, xlab = "λ", ylab = "CV error", legend=false)
    λopt = λs[argmin(errs)]
    λ_mono_ss = [round(λopt, sigdigits = 4), round(log(λopt), sigdigits=4)]
    βhat, yhat = MonotoneSplines.mono_ss(x, y, λopt)
    fig2 = scatter(x, y, label = "obs.")
    scatter!(fig2, x, yhat, label = "mono_ss (λ = $(λ_mono_ss[1]), logλ = $(λ_mono_ss[2]))")
    # ss
    spl = R"smooth.spline($x, $y)"
    λ = rcopy(R"$spl$lambda")
    λ_ss = [round(λ, sigdigits = 4), round(log(λ), sigdigits=4)]
    yhat_ss = rcopy(R"predict($spl, $x)$y")
    scatter!(fig2, x, yhat_ss, label = "ss (λ = $(λ_ss[1]), logλ = $(λ_ss[2]))")
    return plot(fig1, fig2, size = (1240, 420))
end
````

## Growth Curve

````@example monofit
λs = exp.(range(-10, 0, length = 100));
nothing #hide
````

### σ = 3.0

````@example monofit
σ = 3.0
Random.seed!(seed)
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->1/(1-0.42log(z)), xmin = 0, xmax = 10)
scatter(x, y)
scatter!(x0, y0)

demo_mono_ss(x, y, λs)
````

### σ = 2.0

````@example monofit
σ = 2.0
Random.seed!(seed)
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->1/(1-0.42log(z)), xmin = 0, xmax = 10)
scatter(x, y)
scatter!(x0, y0)

demo_mono_ss(x, y, λs)
````

### σ = 0.5

````@example monofit
σ = 0.5
Random.seed!(seed)
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->1/(1-0.42log(z)), xmin = 0, xmax = 10)
scatter(x, y)
scatter!(x0, y0)

demo_mono_ss(x, y, λs)
````

## Logistic Curve

````@example monofit
λs = exp.(range(-10, 0, length = 100));
nothing #hide
````

### σ = 0.2

````@example monofit
σ = 0.2
Random.seed!(seed)
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->exp(z)/(1+exp(z)), xmin = -5, xmax = 5)
scatter(x, y)
scatter!(x0, y0)

demo_mono_ss(x, y, λs)
````

### σ = 1.0

````@example monofit
σ = 1.0
Random.seed!(seed)
x, y, x0, y0 = MonotoneSplines.gen_data(n, σ, z->exp(z)/(1+exp(z)), xmin = -5, xmax = 5)
scatter(x, y)
scatter!(x0, y0)

demo_mono_ss(x, y, λs)
````
