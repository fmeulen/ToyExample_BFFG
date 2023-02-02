wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns  # for softmax
using Random
using DynamicHMC
using UnPack
#using PDMats
using Turing
using StatsPlots # required for Turing plots
using BenchmarkTools
using StaticArrays
using NNlib # for softmax
using DataFrames

import StatsBase.sample

const NUM_HIDDENSTATES = 3
const DIM_COVARIATES = 2
const DIM_RESPONSE = 4

include("lm_funcs.jl")