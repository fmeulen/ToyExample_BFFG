wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns
using FiniteDiff
using TransformVariables, LogDensityProblems, LogDensityProblemsAD, DynamicHMC, TransformedLogDensities, Random
using MCMCDiagnosticTools, DynamicHMC.Diagnostics
using UnPack

struct ObservationTrajectory{S,T}
    X::Vector{S}  # vector of covariates
    Y::Vector{T}  # vector of responses
end
ObservationTrajectory(X) = ObservationTrajectory(X, fill(missing, length(X)))


# kernel K 
Ki(θ,x) = [softmax([0.0, dot(x,θ.γ12), -Inf])' ; softmax([dot(x,θ.γ21), 0.0, dot(x,θ.γ23)])' ; softmax([-Inf, dot(x,θ.γ32), 0])']
# observation kernel

# kernel Λ (observations)
ψ(x) = 2.0*logistic.(cumsum(x)) .- 1.0

function response(Z) 
    λ = ψ(Z)
    [1.0-λ[1] λ[1]; 1.0-λ[2] λ[2]; 1.0-λ[3] λ[3]]
end
Λi(θ) =[ response(θ.Z1), response(θ.Z2), response(θ.Z3), response(θ.Z4)    ]

function generate_track(θ, 𝒪::ObservationTrajectory, Πroot)             # Generate exact track + observations
    X = 𝒪.X
    Λ = Λi(θ)
    uprev = sample(Weights(Πroot))                  # sample x0
    U = [uprev]
    Y = [ [sample(Weights(Λ[i][uprev,:])) for i in eachindex(Λ)] ]
    for i=eachindex(X)
        u = sample(Weights(Ki(θ,X[i])[uprev,:]))         # Generate sample from previous state
        push!(U, u)
        y =  [sample(Weights(Λ[i][u,:])) for i in eachindex(Λ)] 
        push!(Y, y)
        uprev = u
    end
    (U, Y)
end

function h_from_observation(θ, y::Vector)
    U = Λi(θ)
    u = [U[i][:,y[i]] for i in eachindex(y)]  # only those indices where y is not missing, for an index where it is missing we can just send [1;1;1;1]
    u2 = hcat(u...)
    vec(prod(u2, dims=2))
end

#h_from_observation(θ, Y[1])

function normalise!(x)
    s = sum(x)
    x .= x/s
    log(s)
end

function loglik_and_bif(θ, Πroot, 𝒪::ObservationTrajectory)
    @unpack X, Y = 𝒪
    N = length(Y) - 1
    hprev = h_from_observation(θ, Y[N+1])
    H = [hprev]
    loglik = zero(θ[1][1])
    for i=N:-1:1
        h = (Ki(θ,X[i]) * hprev) .* h_from_observation(θ, Y[i])
        c = normalise!(h)
        loglik += c
        pushfirst!(H, ForwardDiff.value.(h))
        hprev = h
    end
    loglik += log(Πroot' * hprev)
    (ll=loglik, H=H)          
end

function loglik(θ, Πroot, 𝒪::ObservationTrajectory)
    @unpack X, Y = 𝒪
    N = length(Y) - 1
    hprev = h_from_observation(θ, Y[N+1])
    loglik = zero(θ[1][1])
    for i=N:-1:1
        h = (Ki(θ,X[i]) * hprev) .* h_from_observation(θ, Y[i])
        c = normalise!(h)
        loglik += c
        hprev = h
    end
    loglik + log(Πroot' * hprev)
end

# loglik for multiple persons
function loglik(θ, Πroot, 𝒪s::Vector)
    ll = zero(θ[1][1])
    for i ∈ eachindex(𝒪s)
        ll += loglik(θ, Πroot, 𝒪s[i])
    end
    ll
end


negloglik(Πroot, 𝒪) = (θ) ->  -loglik_and_bif(θ, Πroot, 𝒪).ll
∇negloglik(Πroot, 𝒪) = (θ) -> ForwardDiff.gradient(negloglik(Πroot, 𝒪), θ)
loglik(Πroot, 𝒪) = (θ) -> loglik(θ, Πroot, 𝒪) 


function guided_track(θ, Πroot, 𝒪, H)# Generate approximate track
    X = 𝒪.X
    N = length(H) - 1
    uprev = sample(Weights(Πroot .* H[1])) # Weighted prior distribution
    uᵒ = [uprev]
    for i=1:N
            p = Ki(θ,X[i])[uprev,:] .* H[i+1]         # Weighted transition density
            u = sample(Weights(p))
            push!(uᵒ, u)
            uprev = u
    end
    uᵒ
end





# True parameter vector
θ0 = ComponentArray(γ12 = rand(2), γ21 = rand(2), γ23 = rand(2), γ32 = rand(2), 
    Z1=rand(Exponential(1.0),3), Z2=rand(Exponential(1.0),3), Z3=rand(Exponential(1.0),3), Z4=rand(Exponential(1.0),3))
Πroot = [1.0, 1.0, 1.0]/3.0

N = 100
X = [rand(2) for i in 1:N]
𝒪 = ObservationTrajectory(X)


# generate track  
U, Y =  generate_track(θ0, 𝒪, Πroot) 
𝒪 = ObservationTrajectory(X,Y)
# backward filter
ll, H = loglik_and_bif(θ0, Πroot, 𝒪)
# sample from conditioned process
Uᵒ = guided_track(θ0, Πroot, 𝒪, H)
# separately compute loglikelihood
loglik(Πroot, 𝒪)(θ0)
𝒪s = [𝒪, 𝒪]
loglik(Πroot, 𝒪s)(θ0)


# plotting 
N = length(Uᵒ)
ts = 0:(N-1)
pl_paths = plot(ts, Uᵒ, label="recovered")
plot!(pl_paths, ts, U, label="latent", linestyle=:dash)
#plot!(pl_paths, ts, ys .+ 1, label="observed")
pl_paths

#----------------------

der_fwdiff = ∇negloglik(Πroot, 𝒪)(θ0)
der_findiff = FiniteDiff.finite_difference_gradient(negloglik(Πroot, 𝒪), θ0)

@show der_fwdiff-der_findiff

#----------------------------------
# compute mle and plot with loglikelihood
# grid = 0:0.01:1
# θgrid = [ComponentArray(p=x, q=θ0.q, r=θ0.r) for x in grid]
# pl_lik = plot(grid, negloglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
# # out = optimize(negloglik(Πroot, ys), 0.0, 1.0)    # box constrained optimization
# # vline!(pl_lik, [out.minimizer], label="mle")
#  vline!(pl_lik, [θ0.p], label="true")

# someθ = ComponentArray(p=0.5, q=0.7, r=0.5)
# negloglik(Πroot, ys)(someθ)

# ∇negloglik(Πroot, ys)(someθ)
# FiniteDiff.finite_difference_gradient(negloglik(Πroot, ys), someθ)

# # ensure domain is ℝ
# negloglik_repam(Πroot, ys)= (θ) -> negloglik(Πroot, ys)(logistic.(θ))

# opt = optimize(negloglik_repam(Πroot, ys), someθ)    # box constrained optimization
# m = logistic.(opt.minimizer)

# ∇negloglik_repam(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik_repam(Πroot, ys), θ)
# ∇negloglik_repam(Πroot, ys)(opt.minimizer)

optimize(negloglik(Πroot, 𝒪), θ) 

# Try DynamicHMC

t = as((γ12=as(Array, 2), γ21=as(Array, 2), γ23=as(Array, 2), γ32=as(Array, 2),
         Z1=as(Array,asℝ₊, 3), Z2=as(Array, asℝ₊, 3), Z3=as(Array, asℝ₊, 3), Z4=as(Array, asℝ₊, 3)  )) 
p = loglik(Πroot, 𝒪)
P = TransformedLogDensity(t, p)
∇P = ADgradient(:ForwardDiff, P);


# one chain
outhmc = mcmc_with_warmup(Random.default_rng(2), ∇P, 100)
ps = outhmc.posterior_matrix

ps_t = transform.(t, eachcol(ps))

l = @layout [a  b;  c d ; e d]
plot(getindex.(getindex.(ps_t,:γ12),1),label="γ12"); 
hline!([θ0.p],label="")
pl_p2 = histogram(getindex.(ps_t,:p),label=""); vline!([θ0.p],label="")
pl_q = plot(getindex.(ps_t,:q),label="q"); hline!([θ0.q],label="")
pl_q2 = histogram(getindex.(ps_t,:q),label=""); vline!([θ0.q],label="")
pl_r = plot(getindex.(ps_t,:r),label="r"); hline!([θ0.r],label="")
pl_r2 = histogram(getindex.(ps_t,:r),label=""); vline!([θ0.r],label="")
plot(pl_p, pl_p2, pl_q, pl_q2, pl_r, pl_r2, layout=l)



ess, R̂ = ess_rhat(stack_posterior_matrices([outhmc]))
@show R̂

# multiple chains
results = [mcmc_with_warmup(Random.default_rng(), ∇P, 1000) for _ in 1:5]

# To get the posterior for ``α``, we need to use the columns of the `posterior_matrix` and
# then transform
posterior = transform.(t, eachcol(pool_posterior_matrices(results)));
ps_t = posterior

ps_t_γ12 = getindex.(ps_t, :γ12)
ps_t_γ23 = getindex.(ps_t, :γ23)
ps_t_Z2 = getindex.(ps_t, :Z2)
plot(getindex.(ps_t_Z2,1))



# Extract the parameter.
posterior_p = first.(posterior);

# check the mean
mean(posterior_p)

# check the effective sample size
ess, R̂ = ess_rhat(stack_posterior_matrices(results))

# NUTS-specific statistics of the first chain
summarize_tree_statistics(results[1].tree_statistics)



#######     TODO: ADD PRIORS, MAKE SENSIBLE TEST DATASET, ALLOW FOR MULTIPLE PERSONS


# 3 state-model with only transitions to neighbours possible, 4 questionaire questions

prior_sample = [ψ(rand(Exponential(1.0),3)) for i in 1:1000]
ps1 = getindex.(prior_sample,1)
ps2 = getindex.(prior_sample,2)
ps3 = getindex.(prior_sample,3)
mean(ps1)
mean(ps2)
mean(ps3)
histogram(ps1)
histogram(ps2)
histogram(ps3)



# vector of covariates
Z =  

# dealing with missing values: if some y[i]==missing 


