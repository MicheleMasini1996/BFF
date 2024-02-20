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

Let us now introduce a problem from scratch. We can define Alice and Bob's measurement operators as projectors with two inputs and two outputs with:
```julia
PA = projector(1,1:2,1:2,full=true)
PB = projector(2,1:2,1:2,full=true)
```
In particular, the first entry is the number of the subsystem (1 for A and 2 for B), the second is number of outputs (1:2 means that it goes from 1 to 2), the third entry is the number of inputs, and finally we impose full=true to force the operators corresponding to the same input to sum up to the identity.

We define the dichotomic observables of Alice and Bob
```julia
A1 = PA[1,1]-PA[2,1]
A2 = PA[1,2]-PA[2,2]
B1 = PB[1,1]-PB[2,1]
B2 = PB[1,2]-PB[2,2]
```

And we define the constraints of our problem. In particular, we force $\langle A_2\otimes B_2\rangle=1$.
```julia
av_eq = [[A2*B2,1]]
```
