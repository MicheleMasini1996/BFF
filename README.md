# BFF
Compute lower bounds on the Von Neumann conditional entropy with NPA as introduced in https://arxiv.org/abs/2106.13692.

We used the code to do NPA from Erik Woodhead (https://github.com/ewoodhead/QuantumNPA.jl) and the modifications done in https://github.com/MicheleMasini1996/QuantumNPA.jl.

Prerequisites:
```julia
using Pkg; Pkg.add(["Combinatorics", "JuMP", "SCS", "BlockDiagonals", "Mosek", "MosekTools", "FastGaussQuadrature", "LinearAlgebra", "Optim"])
```

Moreover, a Mosek license is necessary to use the code.

One can obtain the results for DIQKD studied in https://arxiv.org/abs/2106.13692 using the file "repr-brown.jl" in this way:
```julia
include("repr-brown.jl")
# we can compute the keyrate at η=1 with a standard quantum strategy
keyrate(pi/4,1; m=16) # m is the number of gauss-radau coefficients
# we can also find heuristically an optimal quantum strategy (here we choose η=0.97)
x0=[pi/4,0,pi/2,pi/4,-pi/4,0]
best_qstrat(x0,0.97;level="2+ A B E",m=16) # level refers to the level of the NPA hierarchy
```
