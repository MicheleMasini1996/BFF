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
# we can compute the keyrate at η=1 with a predefined quantum strategy
η=1
θ=pi/4
keyrate(θ,η; m=16) # m is the number of gauss-radau coefficients
# we can also find heuristically an optimal quantum strategy (here we choose η=0.97)
x0=[pi/4,0,pi/2,pi/4,-pi/4,0]
η=0.97
best_qstrat(x0,η;level="2+ A B E",m=16) # level refers to the level of the NPA hierarchy
```

Let us now introduce a problem from scratch. We start including all the functions with
```julia
include("qnpa.jl")
```

We can define Alice and Bob's measurement operators as projectors with two inputs and two outputs with:
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
av_eq = [ [A2*B2, 1] ]
```

Now, let us define an operator constraint on Alice's observables. In particular, we will force anti-commutativity.
```julia
op_eq = [ A1*A2 + A2*A1 ]
```

Finally, we define the operators used to construct the secret key. In this case we extract the key from the first measurement of Alice.
```julia
Ms = [ PA[1,1] PA[2,1] ]
```

We can now compute the conditional entropy $H(A_1|E)$ in three different ways. The slowest method is the general method introduced in the article of Brown Fawzi Fawzi which can be used in this way:
```julia
m=2
level=1 # the level refers to the level of the localizing matrix constructed to implement the operator constraint, the principal moment matrix will be slightly bigger
HAE_simple(Ms, av_eq, level, m; op_eq=op_eq)
```

Second, we can use speed-up number 3 from Remark 2.6 in this way
```julia
m=8
level=1 # the level refers to the level of the localizing matrix constructed to implement the operator constraint, the principal moment matrix will be slightly bigger
HAE_fast1(Ms, av_eq, level, m; op_eq=op_eq)
```

Furthermore, we can use another speed-up. As pointed out at point 1 of Remark 2.6, the operator equalities $Z_{a,i}^* Z_{a,i}\leq\alpha_i$ and $Z_{a,i} Z_{a,i}^* \leq\alpha_i$ do not always improve the results. With the following function, we will not construct them and we will impose only that $\langle Z_{a,i}^* Z_{a,i}\rangle \leq\alpha_i$ and $\langle Z_{a,i} Z_{a,i}^* \rangle \leq\alpha_i$.

```julia
m=8
level="1+A E+A B" 
HAE_fast2(Ms, av_eq, level, m; op_eq=op_eq)
```
