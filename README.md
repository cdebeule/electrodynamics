# Electrodynamics 1

Educational materials for a graduate-level electrodynamics course, including interactive Julia Pluto notebooks.

## Getting Started with Julia and Pluto

In this course, we use [Julia](https://julialang.org), a fast modern programming language well suited for scientific computing, together with [Pluto](https://plutojl.org), a reactive notebook environment for Julia, to visualize and interactively explore electrodynamics concepts.

The steps below are reproduced from the [Pluto install guide](https://plutojl.org/#install).

1. **Install Julia**: download and install from the official site: https://julialang.org

2. **Install Pluto**: open the Julia REPL (Read-Eval-Print Loop, the interactive command line that starts when you launch Julia) and run:
```julia
   using Pkg
   Pkg.add("Pluto")
```

3. **Open a notebook**: launch Pluto by running:
```julia
   using Pluto
   Pluto.run()
```
   This opens the Pluto start page in your browser. From there, you can open a notebook in one of two ways:

   - **Directly from GitHub**: paste the raw file URL into the "Open a notebook" box, for example:

     https://raw.githubusercontent.com/cdebeule/electrodynamocs/main/pluto/two-charge-field.jl
   
      Pluto will download and run it automatically.

   - **From your own computer**: clone or download this repository, then enter the local path to the notebook (e.g. `pluto/two-charge-field.jl`) in the "Open a notebook" box.

## Notebooks

- [two_charge_field.jl](pluto/two-charge-field.jl)
- [laplace.jl](pluto/laplace.jl)
