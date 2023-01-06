wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using Turing
using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays
using StatsFuns
using FiniteDiff
using TransformVariables, LogDensityProblems, LogDensityProblemsAD, DynamicHMC, TransformedLogDensities
using Random
using MCMCDiagnosticTools, DynamicHMC.Diagnostics
using StatsPlots


# func definitions

"""
    generate_track(K, Λ, Πroot, N)

    K: transition matrix of latent chain
    Λ: transition matrix to observations
    Πroot: prior vector on initial state

    returns (xs, ys), where xs is the latent state, and ys the observed value
    both xs and ys are of lenght N+1
"""
function generate_track(K, Λ, Πroot, N)             # Generate exact track + observations
    xprev = sample(Weights(Πroot))                  # sample x0
    xs = [xprev]
    ys = [sample(Weights(Λ[xprev,:]))]
    for i=1:N
        x = sample(Weights(K[xprev,:]))         # Generate sample from previous state
        push!(xs, x)
        y = sample(Weights(Λ[x,:]))             # Generate observation from sample
        push!(ys, y)
        xprev = x
    end
    (xs, ys)
end


"""
    normalise!(x)

    in place normalisation of x such that its elements sum to 1
    returns log(sum(x))
"""
function normalise!(x)
    s = sum(x)
    x .= x/s
    log(s)
end

"""
    loglik_and_bif(θ, Πroot, ys)

    with parameter value θ, prior on initial state Πroot and observation vector ys, compute 
    the backward information filter 

    returns (ll=loglik, h=hs)       
"""
function loglik_and_bif(θ, Πroot, ys)
    N = length(ys) - 1
    K = Ki(θ)
    Λ = Λi(θ)
    hprev = Λ[:,ys[N+1]] 
    # hprev = convert.(ForwardDiff.Dual, Λ[:,ys[N+1]] )    # alternative option to make function AD proof
    hs = [hprev]
    loglik = zero(θ[1])
    for i=N:-1:1
        h = (K * hprev) .* Λ[:,ys[i]]  
        c = normalise!(h)
        loglik += c
#        pushfirst!(hs, h) # not AD-proof
        pushfirst!(hs, ForwardDiff.value.(h))
        
        hprev = h
    end
    loglik += log(Πroot' * hprev)  
    (ll=loglik, h=hs)          
end

"""
    loglik_and_bif(θ, Πroot, ys)

    with parameter value θ, prior on initial state Πroot and observation vector ys, compute 
    the backward information filter 

    returns loglikelihood
"""
function loglik(θ, Πroot, ys) # don't save h functions
    N = length(ys) - 1
    K = Ki(θ)
    Λ = Λi(θ)
    hprev = Λ[:,ys[N+1]] 
    ll = zero(θ[1])
    for i=N:-1:1
        h = (K * hprev) .* Λ[:,ys[i]]  
        c = normalise!(h)
        ll += c
       hprev = h
    end
    ll += log(Πroot' * hprev)
    ll 
end



negloglik(Πroot, ys) = (θ) ->  -loglik_and_bif(θ, Πroot, ys).ll
∇negloglik(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik(Πroot, ys), θ)
loglik(Πroot, ys) = (θ) -> loglik(θ, Πroot, ys) 


"""
    guided_track(K, Πroot, hs)

    K: transition matrix of latent chain
    Πroot: prior vector on initial state
    hs: Doob's h-Transforms

    returns sample of guided process
"""
function guided_track(K, Πroot, hs)# Generate approximate track
    N = length(hs) - 1
    xprev = sample(Weights(Πroot .* hs[1])) # Weighted prior distribution
    xᵒ = [xprev]
    for i=1:N
            p = K[xprev,:] .* hs[i+1]         # Weighted transition density
            x = sample(Weights(p))
            push!(xᵒ, x)
            xprev = x
    end
    xᵒ
end


####################################################################################
# Define the state space
E = [1, 2, 3]

# Length of Markov chain
N = 1000

Ki(θ) = [1.0-0.5θ.p 0.25θ.p   0.25θ.p  ;
         #0.0   1.0-θ.q  θ.q ;  
         0.0 0.7 0.3;
         0.4θ.p   0.3θ.p   1.0-0.7θ.p]

Λi(θ) = [1.0 0.0; 1.0 0.0; θ.r 1.0-θ.r] #0.0 1.0]
Πroot = [1.0, 1.0, 1.0]/3.0

# True parameter vector
θ0 = ComponentArray(p=0.3, q=0.8, r=0.2)
# Hidden Markov chain transition kernel
K = Ki(θ0)
Λ = Λi(θ0)

# generate track, and sample from conditioned process
xs, ys = generate_track(K, Λ, Πroot, N)
(l, hs) = loglik_and_bif(θ0, Πroot, ys)
xstars = guided_track(K, Πroot, hs)

# plotting 
ts = 0:N
pl_paths = plot(ts, xstars, label="recovered")
plot!(pl_paths, ts, xs, label="latent", linestyle=:dash)
plot!(pl_paths, ts, ys .+ 1, label="observed")
pl_paths

# compute mle with q and r fixed at true value and plot with loglikelihood
grid = 0:0.01:1
θgrid = [ComponentArray(p=x, q=θ0.q, r=θ0.r) for x in grid]
pl_lik = plot(grid, loglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
vline!(pl_lik, [θ0.p], label="true")

 # some checks on automatic differentiation
someθ = ComponentArray(p=0.5, q=0.7, r=0.5)
negloglik(Πroot, ys)(someθ)

∇negloglik(Πroot, ys)(someθ)
FiniteDiff.finite_difference_gradient(negloglik(Πroot, ys), someθ)

# ensure domain is ℝ and use optimiser
negloglik_repam(Πroot, ys)= (θ) -> negloglik(Πroot, ys)(logistic.(θ))
opt = optimize(negloglik_repam(Πroot, ys), someθ)    # box constrained optimization
m = logistic.(opt.minimizer)

∇negloglik_repam(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik_repam(Πroot, ys), θ)
∇negloglik_repam(Πroot, ys)(opt.minimizer)  # should be close to zero

#############################################################################
# DynamicHMC (assume uniform distribution on (p,q,r))

logprior() = (θ) -> logpdf(Beta(3.0, 1.0),θ.q)    # just as illustration
p(Πroot, ys) = (θ) -> loglik(Πroot, ys)(θ) + logprior()(θ) 

t = as((p=as_unit_interval , q=as_unit_interval, r=as_unit_interval))
P = TransformedLogDensity(t, p(Πroot, ys))
∇P = ADgradient(:ForwardDiff, P);

@time outhmc = mcmc_with_warmup(Random.default_rng(2), ∇P, 1000);  # one chain

ps = outhmc.posterior_matrix
ps_t = transform.(t, eachcol(ps))

l = @layout [a  b;  c d ; e d]
pl_p = plot(getindex.(ps_t,:p),label="p"); hline!([θ0.p],label="")
pl_p2 = histogram(getindex.(ps_t,:p),label=""); vline!([θ0.p],label="")
pl_q = plot(getindex.(ps_t,:q),label="q"); hline!([θ0.q],label="")
pl_q2 = histogram(getindex.(ps_t,:q),label=""); vline!([θ0.q],label="")
pl_r = plot(getindex.(ps_t,:r),label="r"); hline!([θ0.r],label="")
pl_r2 = histogram(getindex.(ps_t,:r),label=""); vline!([θ0.r],label="")
plot(pl_p, pl_p2, pl_q, pl_q2, pl_r, pl_r2, layout=l)
savefig("dynamic_hmc.png")

# HMC diagnostics
ess, R̂ = ess_rhat(stack_posterior_matrices([outhmc]))
@show R̂


# can also do multiple chains
if false 
    results = [mcmc_with_warmup(Random.default_rng(), ∇P, 1000) for _ in 1:5]

    # To get the posterior for ``α``, we need to use the columns of the `posterior_matrix` and
    # then transform
    posterior = transform.(t, eachcol(pool_posterior_matrices(results)));
    ps_t = posterior

    posterior_p = first.(posterior);
    mean(posterior_p)

    # check the effective sample size
    ess, R̂ = ess_rhat(stack_posterior_matrices(results))

    # NUTS-specific statistics of the first chain
    summarize_tree_statistics(results[1].tree_statistics)
end



################### with Turing.jl (seems somewhat easier)

@model function logtarget(ys, Πroot)
    p ~ Uniform(0.0, 1.0)
    q ~ Beta(3.0, 1.0)
    r ~ Uniform(0.0, 1.0)
    Turing.@addlogprob! loglik(Πroot, ys)(ComponentVector(p=p, q=q, r=r))
end

# multiple samplers to choose from, such as 
sampler = DynamicNUTS() # HMC(0.05, 10);

model = logtarget(ys, Πroot)
@time chain = sample(model, sampler, 1_000; progress=false);
histogram(chain)
savefig("turing.png")

describe(chain)[1]
describe(chain)[2]

# compute map and mle 
map_estimate = optimize(model, MAP())
mle_estimate = optimize(model, MLE())
#coeftable(mle_estimate) # does not work (why???)

# initialise from MLE
@time chain = sample(model, sampler, 1_000, init_params = map_estimate.values.array; progress=false);
plot(chain) 