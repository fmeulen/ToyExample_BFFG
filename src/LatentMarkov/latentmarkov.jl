wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns
using Random
using DynamicHMC
using UnPack
using PDMats
using Turing
using StatsPlots
using BenchmarkTools
using StaticArrays
using NNlib # for softmax
import StatsBase.sample


const NUM_HIDDENSTATES = 3
const DIM_COVARIATES = 2
const DIM_RESPONSE = 4

include("lm_funcs.jl")


########### An example, where data are generated from the model ####################

# True parameter vector
γup = 2.0; γdown = -0.5
γ12 = γ23 = [γup, 0.0]
γ21 = γ32 = [γdown, -0.1]
Z0 = [0.5, 1.0, 1.5]
θ0 = ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ23, γ32 = γ32, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0)

println("true vals", "  ", γup,"  ", γdown,"  ", Z0)

# generate covariates
n = 20 # nr of subjects
T = 15 # nr of times at which we observe
# el1 = intensity, el2 = gender

TX = Union{Missing, SVector{DIM_COVARIATES,Float64}} # indien er missing vals zijn 
TY = Union{Missing, SVector{DIM_RESPONSE, Int64}}

# TX = SVector{2,Float64}
# TY = SVector{DIM_RESPONSE, Int64}

𝒪s = ObservationTrajectory{TX, TY}[]
Us =  Vector{Int64}[]
for i in 1:n
    #local X 
    X = TX[]   # next, we can push! elements to X
    if i ≤ 10 
        for t in 1: T
            push!(X, SA[-0.05*t + 0.2*randn(), 1.0])
        end
    else
        for t in 1: T
            push!(X, SA[-0.05*t + 0.2*randn(), 1.0])
        end
        X[3] = missing
    end
    U, Y =  sample(θ0, X) 
    push!(Us, U)
    YY = TY[]
    push!(YY, missing) 
    for t in  2:T
        push!(YY, Y[t]) 
    end    
    push!(𝒪s, ObservationTrajectory(X, YY))
end

############ use of Turing to sample from the posterior ################

@model function logtarget(𝒪s)
    γ12 ~ filldist(Normal(0,2), DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γ21 ~ filldist(Normal(0,2), DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    Z0 ~ filldist(Exponential(), NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ12, γ32 = γ21, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0), 𝒪s)
end

@model function logtarget_large(𝒪s)
    γ12 ~ filldist(Normal(0,2), DIM_COVARIATES)#MvNormal(fill(0.0, 2), 2.0 * I)
    γ21 ~ filldist(Normal(0,2), DIM_COVARIATES)  #MvNormal(fill(0.0, 2), 2.0 * I)
    
    Z1 ~ filldist(Exponential(), NUM_HIDDENSTATES) 
    Z2 ~ filldist(Exponential(), NUM_HIDDENSTATES) 
    Z3 ~ filldist(Exponential(), NUM_HIDDENSTATES) 
    Z4 ~ filldist(Exponential(), NUM_HIDDENSTATES) 
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ12, γ32 = γ21, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4), 𝒪s)
end


model = logtarget(𝒪s);
model = logtarget_large(𝒪s);

# compute map and mle 
@time map_estimate = optimize(model, MAP())
@time mle_estimate = optimize(model, MLE())

println("true vals", "  ", γ12,"  ", γ21,"  ", Z0)

#sampler = DynamicNUTS() # HMC(0.05, 10);
sampler = NUTS()

@time chain = sample(model, sampler, 1000)#; progress=true);

# plotting 
histogram(chain)
savefig("figs/latentmarkov_histograms.png")
plot(chain)
savefig("figs/latentmarkov_traceplots_histograms.png")

@show chain 
# describe(chain)[1]
# describe(chain)[2]





