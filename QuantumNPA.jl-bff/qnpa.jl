using Base.Iterators
#  flatten, zip (filter?)

using Combinatorics
#  powerset

using JuMP
using SCS

using LinearAlgebra

using SparseArrays
using BlockDiagonals


include("src/operators.jl")
include("src/ops_predefined.jl")

include("src/npa.jl")

import FastGaussQuadrature
include("src/bff.jl")