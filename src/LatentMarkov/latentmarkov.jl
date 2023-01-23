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

θmap =  map_estimate.values

mapallZtoλ(θ) = hcat(mapZtoλ(θ[5:7]), mapZtoλ(θ[8:10]), mapZtoλ(θ[11:13]), mapZtoλ(θ[14:16]))


println("true vals", "  ", γ12,"  ", γ21,"  ", Z0)

#sampler = DynamicNUTS() # HMC(0.05, 10);
sampler = NUTS()

@time chain = sample(model, sampler, 1000)#; progress=true);

sampler =  NUTS(1000, 0.65) 
@time chain = sample(model, sampler, MCMCDistributed(), 1000, 4)#; progress=true);

# plotting 
histogram(chain)
savefig("figs/latentmarkov_histograms.png")
plot(chain)
savefig("figs/latentmarkov_traceplots_histograms.png")

@show chain 

# describe(chain)[1]
# describe(chain)[2]

# vector with posterior means
θpostmean = describe(chain)[1].nt.mean
mapallZtoλ(θpostmean)


chain[:,Symbol("γ12[2]"),1]
chain.value[:,1:7,1]  # first 7 pars, posterior samples

Zs = chain.value[:,5:7,1]
[mapZtoλ(z) for z ∈ eachrow(Zs)]