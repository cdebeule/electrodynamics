# Electrodynamics 1

Educational materials for a graduate-level electrodynamics course, including interactive Julia Pluto notebooks.

## Getting Started with Julia and Pluto

In this course, we use [Julia](https://julialang.org), a fast, modern programming language well suited for scientific computing, together with [Pluto](https://plutojl.org), a reactive notebook environment for Julia, to visualize and interactively explore electrodynamics concepts.

The steps below are reproduced from the [Pluto install guide](https://plutojl.org/#install).

1. **Install Julia**: download and install from the official site: https://julialang.org

2. **Install Pluto**: open the Julia REPL and run:
```julia
   using Pkg
   Pkg.add("Pluto")
```

3. **Launch Pluto**: in the Julia REPL, run:
```julia
   using Pluto
   Pluto.run()
```
   This opens Pluto in your browser, where you can open any `.jl` notebook from the `pluto/` folder.

## Notebooks

- [two_charge_field.jl](pluto/two_charge_field.jl)
- [laplace.jl](pluto/laplace.jl)
