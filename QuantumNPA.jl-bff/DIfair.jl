include("qnpa.jl")
using LinearAlgebra, Optim, Plots

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

# Observable in ZX
O(α) = cos(α).* [1 0; 0 -1]+sin(α).* [0 1; 1 0]
# Projector for outcome a
PO(a,α)=1/2 .* (I(2)+(3-2*a).* O(α))

function A(a,x,η,p; α1=0, α2=pi/2)
    # Alice's POVMs
    A=(2-x).*PO(1,α1)+(x-1).*PO(1,α2)
    A=1/(1-(1-η)*(1-p)^2).*(η*(1-p).*A+(p).*I(2))
    A=(2-a).*A+(a-1).*(I(2)-A)
    return A
end

function B(b,y,η,p; β1=pi/4, β2=-pi/4, β3=0)
    # Bob's POVMs
    if y==3
        B1=PO(1,β3)
    else
        B1=(2-y).*PO(1,β1)+(y-1).*PO(1,β2)
    end
    if b==1
        B=η*(1-p).*B1+(1-η)*p*(1-p).*I(2)
    elseif b==2
        B=η*(1-p).*(I(2)-B1)+(1-η)*p*(1-p).*I(2)
    else
        B=(η*p+(1-η)*(p^2+(1-p)^2)).*I(2)
    end
    return B
end

function pr(a,b,x,y,ηA,ηB,p,v; θ=pi/4, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, β3=0)
    # input-output probability distribution 
    ρ = v.*[cos(θ)^2 0 0 cos(θ)*sin(θ) ; 0 0 0 0; 0 0 0 0; cos(θ)*sin(θ) 0 0 sin(θ)^2] + (1-v)/4 .*I(4)
    tr(ρ*kron(A(a,x,ηA,p; α1=α1, α2=α2),B(b,y,ηB,p; β1=β1, β2=β2)))
end

function hab(θ,ηA,ηB,p,v,α1,β3; q=0)
    # Computes conditional entropy H(A1|B3)
    pab=[pr(1,1,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3) pr(1,2,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3) pr(1,3,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3);
         pr(2,1,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3) pr(2,2,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3) pr(2,3,1,3,ηA,ηB,p,v;θ=θ,α1=α1,β3=β3)]
    pab = transpose([(1-q).*pab[1,:]+q.*pab[2,:] q.*pab[1,:]+(1-q).*pab[2,:]])
    pb = [pab[1,1]+pab[2,1] pab[1,2]+pab[2,2] pab[1,3]+pab[2,3]]
    sum(xlog2x(pab[i,j]) for i in 1:2, j in 1:3)-sum(xlog2x(pb[i]) for i in 1:3)
end

function hae_2323(θ,ηA,ηB,p,v; level="1+A E+A B", m=8, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, β3=0, q=0, localizing=false)
    # Lower bound on the conditional entropy H(A1|E) with 3 outputs on Bob
    # It gives quick results (it imposes only <Z*Z> <= α and <ZZ*> <= α) if localizing=false
    PA = projector(1,1:2,1:2,full=true)
    PB = projector(2,1:3,1:2,full=true)
    constr(a,b,x,y) = [PA[a,x]*PB[b,y],pr(a,b,x,y,ηA,ηB,p,v; θ=θ, α1=α1, α2=α2, β1=β1, β2=β2)]
    av_eq = [ constr(1,1,1,1), constr(1,2,1,1), constr(1,3,1,1), constr(2,1,1,1), constr(2,2,1,1), 
              constr(1,1,1,2), constr(1,2,1,2), constr(2,1,1,2), constr(2,2,1,2), 
              constr(1,1,2,1), constr(1,2,2,1), constr(1,3,2,1), 
              constr(1,1,2,2), constr(1,2,2,2) ]   

    Ms = [(1-q)*PA[1,1]+q*PA[2,1] q*PA[1,1]+(1-q)*PA[2,1]]
    if localizing==true
        return HAE_fast(Ms, av_eq, level, m)
    else
        return HAE_fast3(Ms, av_eq, level, m)
    end
end

function hae(θ,ηA,ηB,p,v; level=2, m=8, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, localizing=false, q=0)
    # case of 2 input 2 output
    PA=projector(1,1:2,1:2,full=true)
    PB=projector(2,1:2,1:2,full=true)
    
    constr(a,b,x,y) = [PA[a,x]*PB[b,y],pr(a,b,x,y,θ,ηA,ηB,p,v; α1=α1, α2=α2, β1=β1, β2=β2)]
    constr2(a,x,y) = [PA[a,x]*PB[2,y],pr(a,2,x,y,θ,ηA,ηB,p,v; α1=α1, α2=α2, β1=β1, β2=β2)+pr(a,3,x,y,θ,ηA,ηB,p,v; α1=α1, α2=α2, β1=β1, β2=β2)]
    av_eq = [ constr(1,1,1,1),constr2(1,1,1),constr(2,1,1,1),
              constr(1,1,1,2),constr(2,1,1,2),
              constr(1,1,2,1),constr2(1,2,1),
              constr(1,1,2,2) ]
    Ms = [ PA[1,1] PA[2,1] ]
    
    if localizing==true
        return HAE_fast1(Ms, av_eq, level, m)
    else
        return HAE_fast2(Ms, av_eq, level, m)
    end
end

function keyrate(θ,ηA,ηB,p; v=1, level="2+A B E", m=15, α1=0, α2=pi/2, β1=pi/4, β2=-pi/4, β3=0, q=0, localizing=false)
    # returns keyrate
    if q<0 || q>=0.5
        return 0
    else
        res = hae_2323(θ,ηA,ηB,p,v; level=level, m=m, α1=α1, α2=α2, β1=β1, β2=β2, q=q, localizing=localizing)-hab(θ,ηA,ηB,p,v,α1,β3; q=q)
        return res
    end
end

function best_qstrat(x0,ηA,ηB,p; v=1, m=15, level="2+A B E")
    # optimizes keyrate over θ,α1,α2,β1,β2,β3,q
    res=Optim.optimize(x->-keyrate(x[1],ηA,ηB,p; v=v, level=level, m=m, α1=x[2], α2=x[3], β1=x[4], β2=x[5], β3=x[6],q=x[7]),x0,Optim.Options(iterations=60))
    output=-Optim.minimum(res)
    output,Optim.minimizer(res)
end

function rand_qstrat(η)
    # randomizes an initial quantum strategy
    x0=[pi/4*rand(),pi*(rand()-0.5),pi*(rand()-0.5),pi*(rand()-0.5),pi*(rand()-0.5),pi*(rand()-0.5)]
    best_qstrat(x0,η)
end