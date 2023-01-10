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

import StatsBase.sample
struct ObservationTrajectory{S,T}
    X::Vector{S}  # vector of covariates (each element of X contains the covariates at a particular time instance)
    Y::Vector{T}  # vector of responses (each element of Y contains a K-vector of responses to the K questions)
end
ObservationTrajectory(X, dimY) = ObservationTrajectory(X, fill(fill(1,dimY), length(X)))  # constructor if only X is given


# transition kernel of the latent chain
Ki(θ,x) = [softmax([0.0, dot(x,θ.γ12), -Inf])' ; softmax([dot(x,θ.γ21), 0.0, dot(x,θ.γ23)])' ; softmax([-Inf, dot(x,θ.γ32), 0])']

# construct transition kernel Λ to observations
# we assume each element of the vector Z is nonnegative. A prior on 
# λ1, λ2, λ3 is formed by setting λi = logistic(cumsum(Z)[i])
# TODO consider better motivated choices

scaledandshifted_logistic(x) = 2.0logistic(x) -1.0 # function that maps [0,∞) to [0,1)
"""
    make matrix [1.0-λ[1] λ[1]; 1.0-λ[2] λ[2]; 1.0-λ[3] λ[3]] (if 3 latent vars)
"""
function response(Z) 
        λ = scaledandshifted_logistic.(cumsum(Z))
        Λ = Matrix{eltype(λ)}(undef , length(λ), 2)  # 2 comes from assuming binary answers to questions
        for k in eachindex(λ)
            Λ[k,:] =[  one(λ[k])-λ[k] λ[k] ]
        end
        Λ            
end

Λi(θ) =[ response(θ.Z1), response(θ.Z2), response(θ.Z3), response(θ.Z4)    ]


sample_observation(Λ, u) =  [sample(Weights(Λ[i][u,:])) for i in eachindex(Λ)] 

"""
    sample(θ, 𝒪::ObservationTrajectory, Πroot)             

    𝒪.X: vector of covariates, say of length n
    
    samples U_1,..., U_n and Y_1,..., Y_n, where 
    U_1 ~ Πroot
    for i ≥ 2 
        U_i | X_{i-1}, U_{i-1} ~ Row_{U_{i-1}} K(θ,X_{i-1})
    (thus, last element of X are not used)

"""
function sample(θ, 𝒪::ObservationTrajectory, Πroot)             # Generate exact track + observations
    X = 𝒪.X
    Λ = Λi(θ)
    uprev = sample(Weights(Πroot))                  # sample x0
    U = [uprev]
    for i in eachindex(X[1:(end-1)])
        u = sample(Weights(Ki(θ,X[i])[uprev,:]))         # Generate sample from previous state
        push!(U, copy(u))
        uprev = u
    end
    Y = [sample_observation(Λ, u) for u ∈ U]
    (U, ObservationTrajectory(𝒪.X, Y))
end


function h_from_observation(θ::Tθ, y::Vector{T}) where {Tθ,T} 
    Λ = Λi(θ)
    a1 = [Λ[i][:,y[i]] for i in eachindex(y)]  # only those indices where y is not missing, for an index where it is missing we can just send [1;1;1;1], or simply define y as such in case of missingness
    out = [prod(first.(a1))]
    K = length(a1[1])
    for k in 2:K
        push!(out,prod(getindex.(a1,k)) )
    end
    out
end

function normalise!(x)
    s = sum(x)
    x .= x/s
    log(s)
end

function loglik_and_bif(θ, Πroot, 𝒪::ObservationTrajectory)
    @unpack X, Y = 𝒪
    N = length(Y) 
    hprev = h_from_observation(θ, Y[N])
    H = [hprev]
    loglik = zero(θ[1][1])
    for i in (N-1):-1:1
        h = (Ki(θ,X[i]) * hprev) .* h_from_observation(θ, Y[i])
        c = normalise!(h)
        loglik += c
        pushfirst!(H, copy(ForwardDiff.value.(h)))
        hprev = h
    end
    loglik += log(dot(hprev, Πroot))
    (ll=loglik, H=H)          
end



# function loglik(θ::Tθ, Πroot::TΠ, 𝒪::ObservationTrajectory) where {Tθ, TΠ}
#     @unpack X, Y = 𝒪
#     N = length(Y) - 1
#     hprev = h_from_observation(θ, Y[N+1])
#     loglik = zero(θ[1][1])
#     for i=N:-1:1
#         h = (Ki(θ,X[i]) * hprev) .* h_from_observation(θ, Y[i])
#         c = normalise!(h)
#         loglik += c
#         hprev = h
#     end
#     loglik + log(Πroot' * hprev)
# end

function loglik(θ::Tθ, Πroot::TΠ, 𝒪::ObservationTrajectory) where {Tθ, TΠ}
    @unpack X, Y = 𝒪
    N = length(Y) 
    h = h_from_observation(θ, Y[N])
    loglik = zero(θ[1][1])
    for i in (N-1):-1:1
        h = (Ki(θ,X[i]) * h) .* h_from_observation(θ, Y[i])
        c = normalise!(h)
        loglik += c
    end
    loglik + log(dot(h, Πroot))
end





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


function sample_guided(θ, Πroot, 𝒪, H)# Generate approximate track
    X = 𝒪.X
    N = length(H) - 1
    uprev = sample(Weights(Πroot .* H[1])) # Weighted prior distribution
    uᵒ = [uprev]
    for i=1:N
            w = Ki(θ,X[i])[uprev,:] .* H[i+1]         # Weighted transition density
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
#Πroot = [1.0, 1.0, 1.0]/3.0
Πroot = [1.0, 0.0, 0.0]

# generate covariates
n = 40 # nr of subjects
T = 30 # nr of times at which we observe

# el1 = intensity, el2 = gender
X = [ [0.05*t + 0.2*randn(), 0.0] for t in 1:T]
dimY = 4
𝒪s = [ObservationTrajectory(X,dimY)]
for i in 2:n
    if i ≤ 10 
        X = [ [0.05*t + 0.2*randn(), 0.0] for t in 1:T]
    else
        X = [ [-0.05*t + 0.2*randn(), 1.0] for t in 1:T]
    end
    push!(𝒪s, ObservationTrajectory(X, dimY))
end

# generate tracks for all individuals
for i in eachindex(𝒪s)
    U, 𝒪 =  sample(θ0, 𝒪s[i], Πroot) 
    𝒪s[i] = 𝒪
end 


######### testing the code ################
# generate track for one person
U, 𝒪 =  sample(θ0, 𝒪s[1], Πroot) 

# backward filter
ll, H = loglik_and_bif(θ0, Πroot, 𝒪)
# sample from conditioned process
Uᵒ = sample_guided(θ0, Πroot, 𝒪, H)
# compute loglikelihood
loglik(Πroot, 𝒪s)(θ0)

# plotting 
N = length(Uᵒ)
ts = 1:N
Uᵒ = sample_guided(θ0, Πroot, 𝒪, H)
pl_paths = plot(ts, Uᵒ, label="recovered")
plot!(pl_paths, ts, U, label="latent", linestyle=:dash)

pl_paths


######### end testing the code ################






#---------------------- check computing times
@time loglik(Πroot, 𝒪s)(θ0);
@time ∇loglik(Πroot, 𝒪s)(θ0);

####### ForwardDiff is faster and allocates less than FiniteDiff ###########
TESTING = false
if TESTING
    using FiniteDiff
    using BenchmarkTools
    ∇loglik_fd(Πroot, 𝒪) = (θ) -> FiniteDiff.finite_difference_gradient(loglik(Πroot, 𝒪), θ)
    @btime ∇loglik_fd(Πroot, 𝒪)(θ0);
    @btime ∇loglik(Πroot, 𝒪)(θ0);
end
##########################################################################
# use of Turing to sample from the posterior

@model function logtarget(𝒪s, Πroot)
    γup ~ Normal(0,3)
    γdown ~ Normal(0,3)
    γ12 = γ23 = [γup, 0.0]
    γ21 = γ32 = [γdown, -0.1]
    Z0 ~ filldist(Exponential(), 3) 
    Turing.@addlogprob! loglik(Πroot, 𝒪s)(ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ23, γ32 = γ32, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0))
end

# @model function logtarget2(𝒪s, Πroot, K)  # K is nr of latent states (turns out that be much slower)
#     γup ~ Normal(0,3)
#     γdown ~ Normal(0,3)
#     γ12 = γ23 = [γup, 0.0]
#     γ21 = γ32 = [γdown, -0.1]
#     r ~ Dirichlet(fill(1,K+1))
#    # r ~ filldist(Gamma(2.0,1.0), K+1)
#     Z0 = cumsum(r)[1:K] #/sum(r)
#     Turing.@addlogprob! loglik(Πroot, 𝒪s)(ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ23, γ32 = γ32, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0))
# end





# multiple samplers to choose from, such as 
sampler = DynamicNUTS() # HMC(0.05, 10);
model = logtarget(𝒪s, Πroot)

# compute map and mle 
@time map_estimate = optimize(model, MAP())
#@time mle_estimate = optimize(model, MLE())

println("true vals", "  ", γup,"  ", γdown,"  ", Z0)

@time chain = sample(model, sampler, 1_000, init_params = map_estimate.values.array; progress=false);
#@time chain = sample(model, sampler, 1_000)#; progress=true);

# plotting 
histogram(chain)
savefig("latentmarkov_histograms.png")
plot(chain)
savefig("latentmarkov_traceplots_histograms.png")

describe(chain)[1]
describe(chain)[2]






# TODO: profiling
# using ProfileView

# ProfileView.@profview loglik(θ0, Πroot, 𝒪s)
# ProfileView.@profview ∇loglik(Πroot, 𝒪)(θ0)

# @code_warntype loglik(θ0, Πroot, 𝒪s[1])
# @code_warntype loglik(θ0, Πroot, 𝒪s)

# y =𝒪s[1].Y[2]
# θ = θ0
# @code_warntype h_from_observation(θ, y)

# @code_warntype ∇loglik(Πroot, 𝒪s[1])(θ0);

#using BenchmarkTools


# l = @layout [a  b;  c d ; e d]
# getindex.(getindex.(ps_t,:γ12),2)
# getindex.(getindex.(ps_t,:Z1),3)
# plot(getindex.(getindex.(ps_t,:γ12),1),label="γ12"); 
# hline!([θ0.p],label="")
# pl_p2 = histogram(getindex.(ps_t,:p),label=""); vline!([θ0.p],label="")
# pl_q = plot(getindex.(ps_t,:q),label="q"); hline!([θ0.q],label="")
# pl_q2 = histogram(getindex.(ps_t,:q),label=""); vline!([θ0.q],label="")
# pl_r = plot(getindex.(ps_t,:r),label="r"); hline!([θ0.r],label="")
# pl_r2 = histogram(getindex.(ps_t,:r),label=""); vline!([θ0.r],label="")
# plot(pl_p, pl_p2, pl_q, pl_q2, pl_r, pl_r2, layout=l)

