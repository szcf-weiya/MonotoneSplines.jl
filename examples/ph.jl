# This section analyzes the polarization hole data using the monotone splines techniques.

using MonotoneSplines
using Plots
using DelimitedFiles

# First of all, we load the data.
current_folder = @__DIR__
data = readdlm(joinpath(current_folder, "ph.dat"));
x = data[:, 1]
y = data[:, 2]
x0 = range(minimum(x), maximum(x), length=500)

# Then we can check how the data looks like
scatter(x, y, label = "")

# ## Monotone Cubic Splines
# Perform the monotone cubic splines with different number of basis functions $J=4, 10$
fit_mcs4 = mono_cs(x, y, 4, increasing = false)
plot!(x0, predict(fit_mcs4, x0), label = "J = 4", legend = :bottomleft)

fit_mcs10 = mono_cs(x, y, 10, increasing = false)
plot!(x0, predict(fit_mcs10, x0), label = "J = 10")
# ## Monotone Smoothing Splines

# Perform smoothing splines
yhat_ss, yhatnew_ss, _, λ = MonotoneSplines.smooth_spline(x, y, x0);
# use the same $\lambda$,
fit_mss = mono_ss(x, y, λ, increasing  = false)
# then plot it
plot!(x0, yhatnew_ss, ls = :dot, label = "Smoothing Spline (λ = $(round(λ, sigdigits = 3)))")
plot!(x0, predict(fit_mss, x0), ls = :solid, label = "Monotone Smoothing Spline (λ = $(round(λ, sigdigits = 3)))")

# ## Monotone smoothing splines with cross-validation

# Alternatively, we can find the optimal tuning parameter $\lambda$ by cross-validation,
λs = exp.(-10:0.2:1)
errs, B, L, J = cv_mono_ss(x, y, λs, increasing = false)
λopt = λs[argmin(errs)]

# Fit with the optimal tuning parameter
fit_mss2 = mono_ss(x, y, λopt, increasing = false)
plot!(x0, predict(fit_mss2, x0), label = "Monotone Smoothing Spline (λ = $(round(λopt, sigdigits = 3)))")

# where the cross-validation error curve is as follows,
scatter(log.(λs), errs, label = "")
