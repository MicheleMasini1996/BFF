include("qnpa.jl")
using LinearAlgebra, Optim

function xlog2x(x)
    if x<0 || x>1
        0
    elseif x==0 || x==1
        0
    else
        -x*log2(x)
    end
end

function hshannon(prob)
    sum(xlog2x.(prob))
end

function hbin(x)
    hshannon((x,1-x))
end

function φ(x)
    hbin(0.5*(1+x))
end

"""
In the following, we will use
    ηA = detection efficiency of Alice
    ηB = detection efficiency of Bob
    p = probability of a dark count per round
    q = probability of Alice of flipping her outcome (noisy preprocessing)
    v = visibility of the state prepared

The initial noiseless state prepared is
    |φ(θ)> = cos(θ)|00> + sin(θ)|11>
and the measurements of Alice and Bob are 
    A1=B1=σz
    Α2=Β2=σx
Bob keeps his 3 outcomes separate, while Alice discards her non-detection outcomes
"""


function A(a,x,η,p)
    # Define Alice's POVMs
    A1=(2-x).*[1 0; 0 0]+(x-1)*0.5.*[1 -1; -1 1]
    A2=(2-x).*[0 0; 0 1]+(x-1)*0.5.*[1 1; 1 1]
    A=1/(1-(1-η)*(1-p)^2).*(η*(1-p).*A1+(p).*I(2))
    A=(2-a).*A+(a-1).*(I(2)-A)
    return A
end

function B(b,y,η,p)
    # Define Bob's POVMs
    B1=(2-y).*[1 0; 0 0]+(y-1)*0.5.*[1 -1; -1 1]
    B2=(2-y).*[0 0; 0 1]+(y-1)*0.5.*[1 1; 1 1]
    if b==1
        B=η*(1-p).*B1+(1-η)*p*(1-p).*I(2)
    elseif b==2
        B=η*(1-p).*B2+(1-η)*p*(1-p).*I(2)
    else
        B=(η*p+(1-η)*(p^2+(1-p)^2)).*I(2)
    end
    return B
end

function pr(a,b,x,y,θ,ηA,ηB,p,v)
    # input-output probability distribution 
    ρ = v.*[cos(θ)^2 0 0 cos(θ)*sin(θ) ; 0 0 0 0; 0 0 0 0; cos(θ)*sin(θ) 0 0 sin(θ)^2] + (1-v)/4 .*I(4)
    tr(ρ*kron(A(a,x,ηA,p),B(b,y,ηB,p)))
end

function hae(θ,ηA,ηB,p,v; level=1, m=5, q=0, localizing=false)
    # Lower bound on the conditional entropy H(A1|E)
    # It returns slow results as it builds localizing matrices for Z*Z<=α and ZZ*<=α
    PA=projector(1,1:2,1:2,full=true)
    PB=projector(2,1:3,1:2,full=true)
    constr(a,b,x,y) = [PA[a,x]*PB[b,y],pr(a,b,x,y,θ,ηA,ηB,p,v)]
    av_eq = [ constr(1,1,1,1), constr(1,2,1,1), constr(2,1,1,1), constr(2,2,1,1), constr(1,3,1,1), constr(1,1,1,2), constr(1,2,1,2), constr(2,1,1,2), constr(1,1,2,1), constr(1,2,2,1), constr(1,1,2,2), constr(1,2,2,2) ]
    op_eq = [ (PA[1,1]-PA[2,1])*(PA[1,2]-PA[2,2])+(PA[1,2]-PA[2,2])*(PA[1,1]-PA[2,1])]
    Ms = [(1-q)*PA[1,1]+q*PA[2,1] (1-q)*PA[2,1]+q*PA[1,1]]
    if localizing==true
        return HAE_fast1(Ms, av_eq, level, m; op_eq=op_eq)
    else
        return HAE_fast2(Ms, av_eq, level, m; op_eq=op_eq)
    end
end

function hab(θ,ηA,ηB,p,v; q=0)
    # Computes conditional entropy H(A1|B1)
    ρ = v.*[cos(θ)^2 0 0 cos(θ)*sin(θ) ; 0 0 0 0; 0 0 0 0; cos(θ)*sin(θ) 0 0 sin(θ)^2] + (1-v)/4 .*I(4)
    pab = [tr(ρ*kron(A(1,1,ηA,p),B(1,1,ηB,p))) tr(ρ*kron(A(1,1,ηA,p),B(2,1,ηB,p))) tr(ρ*kron(A(1,1,ηA,p),B(3,1,ηB,p))) ;
            tr(ρ*kron(A(2,1,ηA,p),B(1,1,ηB,p))) tr(ρ*kron(A(2,1,ηA,p),B(2,1,ηB,p))) tr(ρ*kron(A(2,1,ηA,p),B(3,1,ηB,p))) ]
    pab = transpose([(1-q).*pab[1,:]+q.*pab[2,:] q.*pab[1,:]+(1-q).*pab[2,:]])
    pb = [pab[1,1]+pab[2,1] pab[1,2]+pab[2,2] pab[1,3]+pab[2,3]]
    sum(xlog2x(pab[i,j]) for i in 1:2, j in 1:3)-sum(xlog2x(pb[i]) for i in 1:3)
end

function keyrate(θ,η; v=1, m=15, q=0)
    # returns keyrate as functon of angle θ of partial entanglement and detection efficiency η
    if q<0 || q>0.5
        return 0
    else
        res=hae_faster(θ,1,η,0,v; m=m, q=q)-hab(θ,1,η,0,v; q=q)
        return res
    end
end

function best_qstrat(x0,η; v=1, m=15)
    # optimizes keyrate over partial entanglement angle
    res=Optim.optimize(x->-keyrate(x[1],η; v=v, m=m),x0)
    output=-Optim.minimum(res)
    output,Optim.minimizer(res)
end

function best_nstrat(x0,η; v=1, m=15)
    # optimizes keyrate over partial entanglement angle and noisy preprocessing
    lower_bounds = [0,0]
    upper_bounds = [pi/2,0.5]
    res=Optim.optimize(x->-keyrate(x[1],η; v=v, m=m, q=x[2]),x0,Optim.Options(iterations=100))
    output=-Optim.minimum(res)
    output,Optim.minimizer(res)
end
