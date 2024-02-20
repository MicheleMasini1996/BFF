Moments = Dict{Monomial}{BlockDiagonal}

function sparse_sym_add!(matrix, i, j, val)
    matrix[i, j] += val

    if i != j
        matrix[j, i] += val
    end
end

function sparse_Nt(I, J, V, M, N)
    return SparseMatrixCSC{Float64}{Int}(sparse(I, J, V, M, N))
end

function sparse_sym(N, i, j, val)
    if i == j
        return sparse_Nt([i], [i], [val], N, N)
    else
        return sparse_Nt([i, j], [j, i], [val, val], N, N)
    end
end

function npa_moments_block(operators,cPoly)
    N = length(operators)
    iops = collect(enumerate(operators))
    block = Dict{Monomial,SparseMatrixCSC}()

    for (i, x) in iops
        for (j, y) in iops[i:end]
            p = Polynomial(conj_min(conj(x)*cPoly*y))

            for (c, m) in p
                if !haskey(block, m)
                    block[m] = sparse_sym(N, i, j, c)
                else
                    sparse_sym_add!(block[m], i, j, c)
                end
            end
        end
    end

    return block
end

"""
Construct the NPA moment matrix.

The argument operators can in general be an array of arrays of operators
(blocks), e.g. [[Id], [A1 + B1], [A1 - B1]]. It can also be a simple array of
operators, in which case it is treated the same as an array containing a
single array of operators, e.g., [[Id, A1, A2]]). In either case the return
value is a dictionary with:

  * as keys: monomials obtained by multiplying operators in the same blocks
    together.

  * as values: block-diagonal sparse matrices with coefficients obtained
    from multiplying the input operators together.

When the second argument is not specified, the function will build a 
principal moment matrix. When one includes a polynomial in the second 
argument, the ouput will be the localizing moment matrix of the 
polynomial specified.

"""
function npa_moments(operators, cPoly=Id)
    if isempty(operators)
        return moments
    end

    if first(operators) isa Union{Number,Monomial,Polynomial}
        operators = [operators]
    end

    nblocks = length(operators)
    bsizes = length.(operators)
    blocks = [npa_moments_block(i,cPoly) for i in operators]

    ms = monomials(keys(block) for block in blocks)

    moments = Moments()

    for m in ms
        blocks_m = [(haskey(block, m)
                     ? block[m]
                     : (n -> spzeros(n, n))(bsizes[b]))
                    for (b, block) in enumerate(blocks)]

        moments[m] = BlockDiagonal(blocks_m)
    end

    return moments
end



function SparseArrays.dropzeros!(matrix::BlockDiagonal)
    for blk in blocks(matrix)
        dropzeros!(blk)
    end

    return matrix
end

sp1x1(x) = (iszero(x)
            ? spzeros(Float64, 1, 1)
            : SparseMatrixCSC{Float64,Int}(sparse([1], [1], x)))

"""
Generate the NPA relaxation for a given quantum optimisation problem (an
operator expr whose expectation we want to maximise with the expectation
values of the operators constraints set to zero).
"""
function npa2sdp(expr,
                 level_or_moments;
                 eq=[],
                 ge=[])
    if level_or_moments isa Moments
        moments = level_or_moments
    else
        moments = npa_moments(ops_at_level([expr, eq, ge],
                                           level_or_moments))
    end
    
    # Reduce constraints to canonical form
    expr = conj_min(expr)
    eq = linspace(map(conj_min, eq))
    ge = map(conj_min, ge)

    if haskey(eq, Id)
        @error "Contradiction Id = 0 in equality constraints."
    end

    # Reduce the objective expression, using constraints to eliminate
    # monomials
    expr = reduce_expr(expr, eq)
    moments = deepcopy(moments)

    # Reduce moments using equality constraints.
    for (m0, constraint) in eq
        constraint = copy(constraint)
        q = constraint[m0]
        constraint[m0] = 0

        moment0 = moments[m0]
        delete!(moments, m0)

        for (c, m) in constraint
            moments[m] -= rdiv(c, q) * moment0
        end
    end

    # Reduce inequality constraints then absorb them into the moment matrix.
    # Basically, take the coefficients in the inequalities and add them as
    # 1x1 blocks to the moments.
    ge = [reduce_expr(ineq, eq) for ineq in ge]

    for (m, moment) in moments
        append!(blocks(moment),
                [sp1x1(ineq[m]) for ineq in ge])
    end

    # Remove any zero coefficients that might be stored explicitly in the
    # sparse matrices in the blocks.
    # for matrix in values(moments)
    #    dropzeros!(matrix)
    # end
    
    moments = Moments(m => mat
                      for (m, mat) in moments
                          if !iszero(mat))

    return (expr, moments)
end



function bspzeros(bsizes)
    return BlockDiagonal([spzeros(n, n) for n in bsizes])
end

function Base.zero(bm::BlockDiagonal)
    return bspzeros(first.(blocksizes(bm)))
end

function BlockDiagonals.blocksizes(moments::Moments)
    if isempty(moments)
        return []
    else
        return first.(blocksizes(first(moments)[2]))
    end
end



if !@isdefined(default_solver)
    default_solver = SCS.Optimizer
end

function set_solver!(solver)
    global default_solver = solver
end

function set_verbosity!(model, verbose)
    if !isnothing(verbose)
        (!verbose ? set_silent : unset_silent)(model)
    end
end



function expr2objective(expr, vars)
    return expr[Id] + sum(c*vars[m] for (c, m) in expr if m != Id)
end

"""
Convert moments returned by npa2sdp() to moments in a format used by JuMP.jl
or Convex.jl.
"""
function moments2gamma(moments, vars)
    if isempty(moments)
        return []
    end

    n = nblocks(first(moments)[2])
    gamma = Vector(undef, n)

    for (m, moment) in moments
        var = ((m != Id) ? vars[m] : 1)

        for (b, g) in enumerate(blocks(moment))
            if isassigned(gamma, b)
                gamma[b] += g*var
            else
                gamma[b] = g*var
            end
        end
    end

    return gamma
end



function sdp2jump(expr, moments;
                  goal=:maximise,
                  solver=nothing,
                  verbose=nothing)
    if goal in (:maximise, :maximize, :max)
        maximise = true
        s = 1
    elseif goal in (:minimise, :minimize, :min)
        maximise = false
        s = -1
    end
    
    model = !isnothing(solver) ? Model(solver) : Model()

    Z = [@variable(model, [1:n, 1:n], PSD) for n in blocksizes(moments)]

    objective = (sum(LinearAlgebra.tr(s*G*Z[b])
                     for (b, G) in enumerate(blocks(moments[Id])))
                 + expr[Id])
    
    if maximise
        @objective(model, Min, objective)
    else
        @objective(model, Max, objective)
    end

    for (m, moment) in moments
        if m != Id
            c = expr[m]
            
            @constraint(model,
                        sum(LinearAlgebra.tr(F*Z[b])
                            for (b, F) in enumerate(blocks(moment)))
                        + s*c == 0)
        end
    end

    set_verbosity!(model, verbose)

    return model
end



function npa2jump(expr, level_or_moments;
                  eq=[],
                  ge=[],
                  goal=:maximise,
                  solver=nothing,
                  verbose=nothing)
    (expr, moments) = npa2sdp(expr, level_or_moments, eq=eq, ge=ge)

    model = sdp2jump(expr, moments,
                     goal=goal,
                     solver=solver,
                     verbose=verbose)

    return model
end



function npa_opt(expr, level_or_moments;
                 eq=[],
                 ge=[],
                 goal=:maximise,
                 solver=default_solver,
                 verbose=false)
    model = npa2jump(expr, level_or_moments,
                     eq=eq,
                     ge=ge,
                     goal=goal)

    set_optimizer(model, solver)

    if !verbose
        set_silent(model)
    end

    optimize!(model)

    return objective_value(model)
end



npa_max(expr, level; kw...) = npa_opt(expr, level; goal=:maximise, kw...)
npa_min(expr, level; kw...) = npa_opt(expr, level; goal=:minimise, kw...)


"""
Generate the SDP relaxation of a generic polynomial optimization problem
given equality and inequality constraints. The level can be specified
and it refers to the level of the localizing matrices. The words can 
be defined commuting or non commuting.
"""
function npa_general( obj, level ; 
                    op_eq = 0, 
                    op_ge = 0 ,
                    av_eq = 0,
                    av_ge = 0,
                    show_moments=false,
                    verbose=false)
    obj=Polynomial(obj)
    ops = ops_at_level([obj, av_eq, av_ge, op_ge, op_eq], level)
    pol = 1+sum(op_ge)+sum(op_eq)
    deg = Int(ceil(degree(pol)/2))
    ops_add = ops_at_level([op_ge,op_eq], deg)
    ops_principal = unique([ops_add[o]*ops[p] 
                            for o in 1:length(ops_add) for p in 1:length(ops)])
    
    model = Model(Mosek.Optimizer)

    moments_p = npa_moments(ops_principal)
    mons_p = keys(moments_p)
    
    @variable(model, Γ[mons_p])
    @constraint(model,
                sum(Γ[m].*moments_p[m] for m in mons_p) >= 0,
                PSDCone())
    @constraint(model, Γ[Id]==1)

    if av_eq!=0
        av_eq=[[conj_min(av_eq[x][1]+0),av_eq[x][2]] for x in 1:length(av_eq)]
        [@constraint(model, sum(Γ[m]*av_eq[x][1][m] for m in mons_p) == av_eq[x][2]) for x in 1:length(av_eq) ]
    end

    if av_ge!=0
        av_ge=[[conj_min(av_ge[x][1]),av_ge[x][2]] for x in 1:length(av_ge)]
        [@constraint(model, sum(Γ[m]*av_ge[x][1][m] for m in mons_p) >= av_ge[x][2]) for x in 1:length(av_ge) ]
    end

    if op_eq!=0
        moments_eq = [npa_moments(ops,op_eq[x]) for x in 1:length(op_eq)]
        mons_eq = [keys(moments_eq[x]) for x in 1:length(op_eq)]

        [@constraint(model,
                sum(Γ[m].*moments_eq[x][m] for m in mons_eq[x]) .== 0
                ) for x in 1:length(op_eq)]

    end
    if op_ge!=0
        moments_ge = [npa_moments(ops,op_ge[x]) for x in 1:length(op_ge)]
        mons_ge = [keys(moments_ge[x]) for x in 1:length(op_ge)]
    
        [@constraint(model,
                    sum(Γ[m].*moments_ge[x][m] for m in mons_ge[x]) >= 0,
                    PSDCone()) for x in 1:length(op_ge)]
    end
    obj=conj_min(obj)
    @objective(model, Min, sum(obj[m]*Γ[m] for m in monomials(obj)))
    if !verbose
        set_silent(model)
    end
    
    optimize!(model)
    if show_moments==false
        return objective_value(model)
    else
        if op_ge==0
            return objective_value(model), sum(value(Γ[m])*moments_p[m] for m in mons_p)
        else
            return objective_value(model), sum(value(Γ[m])*moments_p[m] for m in mons_p), 
                [sum(value(Γ[m])*moments_ge[x][m] for m in mons_ge[x]) for x in 1:length(op_ge)]
        end
    end
end


function npa_model(level; obj=0,
                    op_eq = 0, 
                    op_ge = 0 ,
                    av_eq = 0,
                    av_ge = 0,
                    add_ops=0,
                    normalized = 1)

    obj=Polynomial(obj)
    ops = ops_at_level([obj, av_eq, av_ge, op_ge, op_eq, add_ops], level)
    pol = 1+sum(op_ge)+sum(op_eq)
    deg = Int(ceil(degree(pol)/2))
    ops_add = ops_at_level([op_ge,op_eq], deg)
    ops_principal = unique([ops_add[o]*ops[p] 
                            for o in 1:length(ops_add) for p in 1:length(ops)])
                    
    model = Model(Mosek.Optimizer)

    moments_p = npa_moments(ops_principal)
    mons_p = keys(moments_p)
    @variable(model, Γ[mons_p])
    
    @constraint(model,
                sum(Γ[m].*moments_p[m] for m in mons_p) >= 0,
                PSDCone())
    @constraint(model, Γ[Id]==normalized)

    if av_eq!=0
        av_eq=[[conj_min(av_eq[x][1]+0),av_eq[x][2]] for x in 1:length(av_eq)]
        [@constraint(model, sum(Γ[m]*av_eq[x][1][m] for m in mons_p) == av_eq[x][2]) for x in 1:length(av_eq) ]
    end

    if av_ge!=0
        av_ge=[[conj_min(av_ge[x][1]),av_ge[x][2]] for x in 1:length(av_ge)]
        [@constraint(model, sum(Γ[m]*av_ge[x][1][m] for m in mons_p) >= av_ge[x][2]) for x in 1:length(av_ge) ]
    end

    if op_eq!=0
        moments_eq = [npa_moments(ops,op_eq[x]) for x in 1:length(op_eq)]
        mons_eq = [keys(moments_eq[x]) for x in 1:length(op_eq)]

        [@constraint(model,
                sum(Γ[m].*moments_eq[x][m] for m in mons_eq[x]) .== 0
                ) for x in 1:length(op_eq)]

    #    [@constraint(model,
    #            sum(Γ[m].*moments_eq[x][m] for m in mons_eq[x]) <= 0,
    #            PSDCone()) for x in 1:length(op_eq)]
    end
    if op_ge!=0
        moments_ge = [npa_moments(ops,op_ge[x]) for x in 1:length(op_ge)]
        mons_ge = [keys(moments_ge[x]) for x in 1:length(op_ge)]
    
        [@constraint(model,
                    sum(Γ[m].*moments_ge[x][m] for m in mons_ge[x]) >= 0,
                    PSDCone()) for x in 1:length(op_ge)]
    end
    if obj!=0
        obj=conj_min(obj)
        @objective(model, Min, sum(obj[m]*Γ[m] for m in monomials(obj)))
    end
    return model, Γ, mons_p
end