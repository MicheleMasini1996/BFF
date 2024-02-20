using Mosek, MosekTools

function gaussradau(m)
    x, v = FastGaussQuadrature.gaussradau(m);
    t = 0.5*(1 .- x);
    w = 0.5*v;

    return (t, w)
end

const ln2 = log(2)

zbff_op(M, Z, Zc, t) = M*(Z + Zc + (1 - t)*Zc*Z) + t*Z*Zc

function HAE_simple(Ms, av_eq, level, m; op_eq=0)
    ts, ws = gaussradau(m)

    # Drop the first one 
    ts = ts[2:end]
    ws = ws[2:end]
    α = [3/2*max(1/ts[i],1/(1-ts[i])) for i in 1:(m-1)]

    Zs = zbff(5, 1:(length(Ms)*(m-1))) 
    Zcs = conj(Zs)
    Zs = reshape(Zs,length(Ms),m-1)
    Zcs = reshape(Zcs,length(Ms),m-1)

    result = 0
    cm=0
    ge=[]

    for i in 1:(m-1)
        for a in 1:length(Ms)
            append!(ge, [Id * α[i] - Zcs[a, i] * Zs[a, i], Id * α[i] - Zs[a, i] * Zcs[a, i]])
        end
    end
    

    O = sum(ws[i]/ts[i]/ln2*zbff_op(Ms[a],Zs[a,i],Zcs[a,i],ts[i]) for i in 1:(m-1), a in 1:length(Ms))
    cm = sum(ws[i]/ts[i]/ln2 for i in 1:(m-1))

    obj = npa_general(O,level; op_eq=op_eq, op_ge=ge, av_eq=av_eq)
    result = cm + obj   

    return result
end

function HAE_fast1(Ms, av_eq, level, m; op_eq=0)
    # get faster lower bound by breaking the optimization into m problems
    # we use localizing matrices for Z*Z<=α and ZZ*<=α 
    ts, ws = gaussradau(m)

    # Drop the first one 
    ts = ts[2:end]
    ws = ws[2:end]
    α = [3/2*max(1/ts[i],1/(1-ts[i])) for i in 1:(m-1)]

    Zs = zbff(5, 1:(length(Ms)*(m-1))) 
    Zcs = conj(Zs)
    Zs = reshape(Zs,length(Ms),m-1)
    Zcs = reshape(Zcs,length(Ms),m-1)    

    sec = 0
    cm = sum(ws[i]/ts[i]/ln2 for i in 1:(m-1))
    for i in 1:(m-1)
        O = ws[i]/ts[i]/ln2*sum(zbff_op(Ms[a],Zs[a,i],Zcs[a,i],ts[i]) for a in 1:length(Ms))
        ge=[]
        for a in 1:length(Ms)
            append!(ge, [Id * α[i] - Zcs[a,i] * Zs[a,i], Id * α[i] - Zs[a,i] * Zcs[a,i]])
        end
        obj = npa_general(O,level; op_eq=op_eq, op_ge=ge, av_eq=av_eq)
        sec = sec+obj
    end
    result = cm + sec   

    return result
end

function HAE_fast2(Ms, av_eq, level, m; op_eq=0, normalized=1)
    # get lower bound without constraining only mean value for Z*Z<=α and ZZ*<=α
    # and preparing only once the SDP model to solve
    ts, ws = gaussradau(m)

    # Drop the first one 
    ts = ts[2:end]
    ws = ws[2:end]
    α = [3/2*max(1/ts[i],1/(1-ts[i])) for i in 1:(m-1)]

    Zs = zbff(5, 1:length(Ms)) 
    Zcs = conj(Zs)
    add_ops = [Zs, Zcs]

    ge=[]
    for a in 1:length(Ms)
        append!(ge, [[- Zcs[a] * Zs[a], -α[1] ]]) 
        append!(ge, [[- Zs[a] * Zcs[a], -α[1] ]])
    end

    model,Γ,mons = npa_model(level; op_eq=op_eq, av_eq=av_eq, add_ops=add_ops, normalized=normalized)
    @constraint(model, c[a=1:2*length(Ms)], sum(Γ[m]*conj_min(ge[a][1][m]) for m in mons) >= ge[a][2])

    sec = 0
    cm = sum(ws[i]/ts[i]/ln2 for i in 1:(m-1))
    for i in 1:(m-1)
        for a in 1:length(Ms)
            set_normalized_rhs(c[a], -α[i])
        end
        O = ws[i]/ts[i]/ln2*sum(zbff_op(Ms[a],Zs[a],Zcs[a],ts[i]) for a in 1:length(Ms)) 
        O = conj_min(O)    
        @objective(model, Min, sum(O[m]*Γ[m] for m in monomials(O)))
        set_silent(model)
        optimize!(model)
        obj = objective_value(model)
        sec = sec+obj
    end
    result = cm + sec  
    
    return result
end