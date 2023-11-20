Rhome = get(ENV, "R_HOME", "")
if Rhome == "*"
    try
        import Conda
        # If using Conda's R, install R package automatically.
        Conda.add("r-fda")
        Conda.add("r-splines")
        Conda.add("r-lsei")
        @info "R packages are installed successfully."
    catch e
        @warn """
        Fail to automatically install dependent R packages in Conda due to $e. 
        
        Please fix the error message first and then reinstall via

        ```julia
        Conda.add("r-fda")
        Conda.add("r-splines")
        Conda.add("r-lsei")
        ```
        
        Alternatively, you can consider using system R, and install the packages via the standard way `install.packages(..)`
        """
    end
else
    @info "You are not using Conda's R, please install R packages by yourself."
end