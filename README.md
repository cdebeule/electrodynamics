# Electrodynamics 1

Educational materials for a graduate-level electrodynamics course, including interactive Julia Pluto notebooks.

## Getting Started with Julia and Pluto

1. **Install Julia**: download and install from the official site: https://julialang.org/downloads/

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

New to Julia and Pluto? The [Julia documentation](https://docs.julialang.org) and [Pluto.jl website](https://plutojl.org) are good places to start.

## Notebooks

- [two_charge_field.jl](pluto/two_charge_field.jl)
- [laplace.jl](pluto/laplace.jl)
