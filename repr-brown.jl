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

function hcond(pab,pb)
    hshannon(pab)-hshannon(pb)
end

"""
In the following, we will use
    η = detection efficiency of Alice and Bob
    q = probability of Alice flipping her outcome (noisy preprocessing)
    v = visibility of the state prepared

The initial noiseless state prepared is
    |φ(θ)> = cos(θ)|00> + sin(θ)|11>
and the measurements of Alice and Bob are 
    Ai=cos(αi)σz+sin(αi)σx
    Βi=cos(βi)σz+sin(βi)σx
Alice and Bob map their non-detection outcome to one of the other outcomes
except during key generation rounds, where Bob keeps his 3 outcomes separate
"""


function A(a,x,η; α1=0, α2=pi/2)
    # Alice's POVMs
    A(α) = cos(α).* [1 0; 0 -1]+sin(α).* [0 1; 1 0]
    A = (2-x).* A(α1) + (x-1).* A(α2)
    A=1/2 .* (I(2)+(3-2*a).* A)
    A=(2-a).* (η.* A+(1-η).* I(2))+(a-1).*(η.* A)
    return A
end

function B(b,y,η; β1=pi/4, β2=-pi/4)
    # Bob's POVMs
    B(β) = cos(β).* [1 0; 0 -1]+sin(β).* [0 1; 1 0]
    B = (2-y).* B(β1) + (y-1).* B(β2)
    B=1/2 .* (I(2)+(3-2*b).* B)
    B=(2-b).* (η.* B+(1-η).* I(2))+(b-1).*(η.* B)
    return B
end

function pr(a,b,x,y,θ,η,v; α1=0, α2=pi/2, β1=pi/4, β2=-pi/4)
    # compute input-output probability distribution p(a,b|x,y)
    ρ = v.*[cos(θ)^2 0 0 cos(θ)*sin(θ) ; 0 0 0 0; 0 0 0 0; cos(θ)*sin(θ) 0 0 sin(θ)^2] + (1-v)/4 .*I(4)
    tr(ρ*kron(A(a,x,η; α1=α1, α2=α2),B(b,y,η; β1=β1, β2=β2)))
end

function pab13(a,b,η; θ=pi/4, v=1, α1=0, β3=0)
    ρ = v.*[cos(θ)^2 0 0 cos(θ)*sin(θ) ; 0 0 0 0; 0 0 0 0; cos(θ)*sin(θ) 0 0 sin(θ)^2] + (1-v)/4 .*I(4)
    if b==3
        return (1-η)*tr(ρ*kron(A(a,1,η; α1=α1),I(2)))
    end
    B = cos(β3).* [1 0; 0 -1]+sin(β3).* [0 1; 1 0]
    B=η*1/2 .* (I(2)+(3-2*b).* B)
    tr(ρ*kron(A(a,1,η; α1=α1),B))
end

function hab(θ,η,v)
    # conditional entropy H(A1|B3) with α1=β3=0
    η/2*(1+v*cos(2*θ))*hbin(η/2*(1-v^2)/(1+v*cos(2*θ)))+
            η/2*(1-v*cos(2*θ))*hbin(η*(1-1/2*(1-v^2)/(1-v*cos(2*θ))))+
                (1-η)*hbin(η/2*(1-v*cos(2*θ)))
end

function hab(θ,η,v,α1,β3)
    # conditional entropy H(A1|B3) 
    Pa1b3=(pab13(1,1,η;θ=θ,v=v,α1=α1,β3=β3),pab13(1,2,η;θ=θ,v=v,α1=α1,β3=β3),pab13(1,3,η;θ=θ,v=v,α1=α1,β3=β3),pab13(2,1,η;θ=θ,v=v,α1=α1,β3=β3),pab13(2,2,η;θ=θ,v=v,α1=α1,β3=β3),pab13(2,3,η;θ=θ,v=v,α1=α1,β3=β3))
    Pb3=(Pa1b3[1]+Pa1b3[4],Pa1b3[2]+Pa1b3[5],Pa1b3[3]+Pa1b3[6])
    hcond(Pa1b3,Pb3)
end

function hae(θ,η,v; level=2, m=8, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, localizing=false)
    # conditional entropy H(A1|E)
    PA=projector(1,1:2,1:2,full=true)
    PB=projector(2,1:2,1:2,full=true)
    
    constr(a,b,x,y) = [PA[a,x]*PB[b,y],pr(a,b,x,y,θ,η,v; α1=α1, α2=α2, β1=β1, β2=β2)]
    av_eq = [ constr(1,1,1,1),constr(1,2,1,1),constr(2,1,1,1),
              constr(1,1,1,2),constr(2,1,1,2),
              constr(1,1,2,1),constr(1,2,2,1),
              constr(1,1,2,2) ]
    Ms = [ PA[1,1] PA[2,1] ]
    
    if localizing==true
        return HAE_fast1(Ms, av_eq, level, m)
    else
        return HAE_fast2(Ms, av_eq, level, m)
    end
end

function keyrate(θ,η; v=1, level="2+A E+A B+A B E", m=8, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, β3=0)
    # computes keyrate H(A1|E)-H(A1|B3)
    en=hae(θ,η,v; level=level, m=m, α1=α1, α2=α2, β1=β1, β2=β2)
    er=hab(θ,η,v,α1,β3)
    return en-er,en,er
end

function best_qstrat(x0,η; level="1+A E+A B", v=1, m=8)
    # optimizes keyrate over quantum strategies (no noisy preprocessing)
    res=Optim.optimize(x->-keyrate(x[1],η; v=v, level=level, m=m, α1=x[2], α2=x[3], β1=x[4], β2=x[5], β3=x[6])[1],x0,Optim.Options(iterations=100))
    output=-Optim.minimum(res)
    output,Optim.minimizer(res)
end

function rand_qstrat(η)
    # optimizes keyrate starting from a random quantum strategy
    x0=[pi/4*rand(),pi*(rand()-0.5),pi*(rand()-0.5),pi*(rand()-0.5)]
    best_qstrat(x0,η)
end
