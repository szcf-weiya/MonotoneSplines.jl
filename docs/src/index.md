# MonotoneSplines.jl Documentation

*MonotoneSplines.jl* is a Julia package for monotone splines, which impose a monotonicity constraint on the smoothing splines. 

```math
\underset{\color{red}{f\textbf{ is monotonic}}}{\arg\min} \sum_{i=1}^n\left\{y_i-f(x_i)\right\}^2 + \lambda \int \left\{f''(t)\right\}^2dt\,,
```

where $f$ is formed with B-spline basis $f(x) = \sum_{j=1}^J\gamma_j B_j(x)$. A sufficient condition for $f$ to be monotonic is $\gamma_1,\ldots,\gamma_J$ is monotonic. With matrix notation ${\mathbf y} = [y_1,\ldots, y_n], {\mathbf B}_{ij} = B_j(x_i), {\boldsymbol\Omega}_{ij} = \int B_i''(s)B_j''(s)ds$, the problem can be rewritten as

```math
\begin{aligned}
\underset{\gamma}{\arg\min} & \Vert {\mathbf y} - {\mathbf B} \gamma\Vert_2^2 + \lambda \gamma^T\boldsymbol\Omega\gamma\\
\text{subject to } & \alpha \gamma_1 \le \alpha \gamma_2\le \cdots \le \alpha\gamma_J\,,
\end{aligned}
```

where $\alpha=1$ implies non-decreasing and $\alpha=-1$ indicates non-increasing.

The package provides two algorithms (frameworks) for fitting the monotone splines.

- Convert the problem into a classical convex second-order cone optimization problem. There are many mature existing optimization toolboxes can be used, such as [ECOS.jl](https://github.com/jump-dev/ECOS.jl).
- Approximate the solution with an Multi-Layer Perceptrons (MLP) generator, using the powerful representation ability of neural network.

Particularly, the second approach can achieve good approximations and it can save much time by avoiding repeating to run the optimization problems of the first approach when we conduct bootstrap to estimate the confidence band. 

We do not *reinvent the wheel*. Instead, we fully take advantage of the existing widely-used implementations in other programming languages with the help of the flexible integration feature of Julia. For example, the package adopts the calculation of B-splines from R's `splines` package via [RCall.jl](https://github.com/JuliaInterop/RCall.jl), and provides the PyTorch deep learning backend via [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) as an alternative to the pure-Julia deep learning framework [Flux.jl](https://github.com/FluxML/Flux.jl).