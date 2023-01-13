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

struct ObservationTrajectory{S,T}
    X::Vector{S}  # vector of covariates (each element of X contains the covariates at a particular time instance)
    Y::Vector{T}  # vector of responses (each element of Y contains a K-vector of responses to the K questions)
end
#ObservationTrajectory(X, dimY) = ObservationTrajectory(X, fill(fill(1,dimY), length(X)))  # constructor if only X is given

ObservationTrajectory(X, dimY) = ObservationTrajectory(X, fill(SA[1,1,1,1], length(X)))  # constructor if only X is given


# transition kernel of the latent chain assuming 3 latent states
#Ki(θ,x) = [StatsFuns.softmax([0.0, dot(x,θ.γ12), -Inf])' ; StatsFuns.softmax([dot(x,θ.γ21), 0.0, dot(x,θ.γ23)])' ; StatsFuns.softmax([-Inf, dot(x,θ.γ32), 0])']
# can also be done with StaticArrays
Ki(θ,x)= NNlib.softmax([0.0 dot(x,θ.γ12) -Inf; dot(x,θ.γ21) 0.0 dot(x,θ.γ23) ; -Inf dot(x,θ.γ32) 0];dims=2)  # slightly faster, though almost double allocation
 
 
scaledandshifted_logistic(x) = 2.0logistic(x) -1.0 # function that maps [0,∞) to [0,1)

"""
    response(Z) 
    
    make matrix [1.0-λ[1] λ[1]; 1.0-λ[2] λ[2]; 1.0-λ[3] λ[3]] (if 3 latent vars)

    # construct transition kernel Λ to observations
    # λ1, λ2, λ3 is formed by setting λi = logistic(cumsum(Z)[i])
"""
function response(Z) 
        λ = scaledandshifted_logistic.(cumsum(Z))
        # Λ = Matrix{eltype(λ)}(undef , length(λ), 2)  # 2 comes from assuming binary answers to questions
        # for k in eachindex(λ)
        #     Λ[k,:] =[  one(λ[k])-λ[k] λ[k] ]
        # end
        # Λ            
        SA[ one(λ[1])-λ[1] λ[1];  one(λ[2])-λ[2] λ[2];  one(λ[3])-λ[3] λ[3]]
end

Λi(θ) = SA[ response(θ.Z1), response(θ.Z2), response(θ.Z3), response(θ.Z4)    ]    # assume 4 questions


#sample_observation(Λ, u) =  [sample(Weights(Λ[i][u,:])) for i in eachindex(Λ)] # sample Y | U
sample_observation(Λ, u) =  SA[sample(Weights(Λ[1][u,:])), sample(Weights(Λ[2][u,:])), sample(Weights(Λ[3][u,:])), sample(Weights(Λ[4][u,:])) ] # sample Y | U

"""
    sample(θ, 𝒪::ObservationTrajectory, Πroot)             

    𝒪.X: vector of covariates, say of length n
    
    samples U_1,..., U_n and Y_1,..., Y_n, where 
    U_1 ~ Πroot
    for i ≥ 2 
        U_i | X_{i}, U_{i-1} ~ Row_{U_{i-1}} K(θ,X_{i})
    (thus, last element of X are not used)

"""
function sample(θ, 𝒪::ObservationTrajectory, Πroot)             # Generate exact track + observations
    X = 𝒪.X
    Λ = Λi(θ)
    uprev = sample(Weights(Πroot))                  # sample u1 (possibly depending on X[1])
    U = [uprev]
    for i in eachindex(X[2:end])
        u = sample(Weights(Ki(θ,X[i])[uprev,:]))         # Generate sample from previous state
        push!(U, copy(u))
        uprev = u
    end
    Y = [sample_observation(Λ, u) for u ∈ U]
    (U, ObservationTrajectory(𝒪.X, Y))
end


function h_from_observation(θ::Tθ, y::T) where {Tθ,T} 
    Λ = Λi(θ)
    # a1 = [Λ[i][:,y[i]] for i in eachindex(y)]  # only those indices where y is not missing, for an index where it is missing we can just send [1;1;1;1], or simply define y as such in case of missingness
    # out = [prod(first.(a1))]
    # K = length(a1[1])
    # for k in 2:K
    #     push!(out,prod(getindex.(a1,k)) )
    # end
    # out
    Λ[1][:,y[1]] .* Λ[2][:,y[2]] .* Λ[3][:,y[3]] .* Λ[4][:,y[4]]
end

function normalise!(x)
    s = sum(x)
    log(s)
end


function loglik_and_bif(θ, Πroot, 𝒪::ObservationTrajectory)
    @unpack X, Y = 𝒪
    N = length(Y) 
    hprev = h_from_observation(θ, Y[N])
    H = [hprev]
    loglik = zero(θ[1][1])
    for i in N:-1:2
        h = (Ki(θ,X[i]) * hprev) .* h_from_observation(θ, Y[i-1])
        c = normalise!(h)
        loglik += c
        pushfirst!(H, copy(ForwardDiff.value.(h)))
        hprev = h
    end
    loglik += log(dot(hprev, Πroot))
    (ll=loglik, H=H)          
end

function loglik(θ::Tθ, Πroot::TΠ, 𝒪::ObservationTrajectory) where {Tθ, TΠ}
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N])
    loglik = zero(θ[1][1])
    for i in N:-1:2
        K = Ki(θ,X[i]) 
       # K = @SMatrix ones(3,3)
        h = (K * h) .* h_from_observation(θ, Y[i-1])
        #@show typeof(h)
        c = normalise!(h)
        loglik += c
    end
    loglik + log(dot(h, Πroot))
end

# to do: make Πroot depend on X[1]

# loglik for multiple persons
function loglik(θ, Πroot, 𝒪s::Vector)
    ll = zero(θ[1][1])
    for i ∈ eachindex(𝒪s)
        ll += loglik(θ, Πroot, 𝒪s[i])
    end
    ll 
end

loglik(Πroot, 𝒪) = (θ) -> loglik(θ, Πroot, 𝒪) 

∇loglik(Πroot, 𝒪) = (θ) -> ForwardDiff.gradient(loglik(Πroot, 𝒪), θ)

# check
function sample_guided(θ, Πroot, 𝒪, H)# Generate approximate track
    X = 𝒪.X
    N = length(H) # check -1?
    uprev = sample(Weights(Πroot .* H[1])) # Weighted prior distribution
    uᵒ = [uprev]
    for i=2:N
            w = Ki(θ,X[i])[uprev,:] .* H[i]         # Weighted transition density
            u = sample(Weights(w))
            push!(uᵒ, u)
            uprev = u
    end
    uᵒ
end


########### An example, where data are generated from the model ####################

# True parameter vector
γup = 2.0; γdown = -0.5
γ12 = γ23 = [γup, 0.0]
γ21 = γ32 = [γdown, -0.1]
Z0 = [0.5, 1.0, 1.5]
θ0 = ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ23, γ32 = γ32, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0)

println("true vals", "  ", γup,"  ", γdown,"  ", Z0)

# Prior on root node
#Πroot = SA[1.0, 1.0, 1.0]/3.0
Πroot = SA[1.0, 0.0, 0.0]

# generate covariates
n = 40 # nr of subjects
T = 30 # nr of times at which we observe

# el1 = intensity, el2 = gender
X = [ SA[0.05*t + 0.2*randn(), 0.0] for t in 1:T]
dimY = 4
𝒪s = [ObservationTrajectory(X,dimY)]
for i in 2:n
    local X 
    if i ≤ 10 
        X = [ SA[0.05*t + 0.2*randn(), 0.0] for t in 1:T]
    else
        X = [ SA[-0.05*t + 0.2*randn(), 1.0] for t in 1:T]
    end
    push!(𝒪s, ObservationTrajectory(X, dimY))
end

# generate tracks for all individuals
for i in eachindex(𝒪s)
    U, 𝒪 =  sample(θ0, 𝒪s[i], Πroot) 
    𝒪s[i] = 𝒪
end 

# use of Turing to sample from the posterior


@model function logtarget(𝒪s, Πroot)
    γ12 ~ filldist(Normal(0,2),2)#MvNormal(fill(0.0, 2), 2.0 * I)
    γ21  ~ filldist(Normal(0,2),2)  #MvNormal(fill(0.0, 2), 2.0 * I)
    Z0 ~ filldist(Exponential(), 3) 
    #Turing.@addlogprob! loglik(Πroot, 𝒪s)(θ)
    Turing.@addlogprob! loglik(ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ12, γ32 = γ21, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0), Πroot, 𝒪s)
end


model = logtarget(𝒪s, Πroot)

# compute map and mle 
@time map_estimate = optimize(model, MAP())
@time mle_estimate = optimize(model, MLE())

println("true vals", "  ", γ12,"  ", γ21,"  ", Z0)

#sampler = DynamicNUTS() # HMC(0.05, 10);
sampler = NUTS()


#@time chain = sample(model, sampler, 1_00, init_params = map_estimate.values.array; progress=false);
@time chain = sample(model, sampler, 1000)#; progress=true);

# plotting 
histogram(chain)
savefig("latentmarkov_histograms.png")
plot(chain)
savefig("latentmarkov_traceplots_histograms.png")

@show chain 
# describe(chain)[1]
# describe(chain)[2]





