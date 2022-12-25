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

# func definitions
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

function normalise!(x)
    s = sum(x)
    x .= x/s
    log(s)
end

function loglik_and_bif(θ, Πroot, ys)
    N = length(ys) - 1
    K = Ki(θ)
    Λ = Λi(θ)
    hprev = Λ[:,ys[N+1]] 
    # hprev = convert.(ForwardDiff.Dual, Λ[:,ys[N+1]] )    
    hs = [hprev]
    loglik = zero(θ[1])
    for i=N:-1:1
        h = (K * hprev) .* Λ[:,ys[i]]  
        c = normalise!(h)
        loglik += c
#        pushfirst!(hs, h)
        pushfirst!(hs, ForwardDiff.value.(h))
        
        hprev = h
    end
    loglik += log(Πroot' * hprev)
    (ll=loglik, h=hs)          
end

negloglik(Πroot, ys) = (θ) ->  -loglik_and_bif(θ, Πroot, ys).ll
∇negloglik(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik(Πroot, ys), θ)

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

# Define the state space
E = [1, 2, 3]

# Length of Markov chain
N = 1000

Ki(θ) = [1.0-0.5θ.p 0.25θ.p   0.25θ.p  ;
         0   1.0-θ.q  θ.q ;  
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

# compute mle and plot with loglikelihood
grid = 0:0.01:1
θgrid = [ComponentArray(p=x, q=θ0.q, r=θ0.r) for x in grid]
pl_lik = plot(grid, negloglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
# out = optimize(negloglik(Πroot, ys), 0.0, 1.0)    # box constrained optimization
# vline!(pl_lik, [out.minimizer], label="mle")
 vline!(pl_lik, [θ0.p], label="true")

someθ = ComponentArray(p=0.5, q=0.7, r=0.5)
negloglik(Πroot, ys)(someθ)

∇negloglik(Πroot, ys)(someθ)
FiniteDiff.finite_difference_gradient(negloglik(Πroot, ys), someθ)

# ensure domain is ℝ
negloglik_repam(Πroot, ys)= (θ) -> negloglik(Πroot, ys)(logistic.(θ))

opt = optimize(negloglik_repam(Πroot, ys), someθ)    # box constrained optimization
m = logistic.(opt.minimizer)

∇negloglik_repam(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik_repam(Πroot, ys), θ)
∇negloglik_repam(Πroot, ys)(opt.minimizer)

# Try DynamicHMC
loglik(Πroot, ys) = (θ) -> -negloglik(Πroot, ys)(θ) 

p = loglik(Πroot, ys)

t = as((p=as_unit_interval , q=as_unit_interval, r=as_unit_interval))
P = TransformedLogDensity(t, p)
@assert P.log_density_function(someθ) ==p(someθ)

∇P = ADgradient(:ForwardDiff, P);

# one chain
outhmc = mcmc_with_warmup(Random.default_rng(2), ∇P, 5000)
ps = outhmc.posterior_matrix

ps_t = transform.(t, eachcol(ps))

l = @layout [a ; b; c]
pl_p = plot(getindex.(ps_t,:p),label="p"); hline!([θ0.p],label="")
pl_q = plot(getindex.(ps_t,:q),label="q"); hline!([θ0.q],label="")
pl_r = plot(getindex.(ps_t,:r),label="r"); hline!([θ0.r],label="")
plot(pl_p, pl_q, pl_r, layout=l)


ess, R̂ = ess_rhat(stack_posterior_matrices([outhmc]))
@show R̂

# multiple chains
results = [mcmc_with_warmup(Random.default_rng(), ∇P, 1000) for _ in 1:5]

# To get the posterior for ``α``, we need to use the columns of the `posterior_matrix` and
# then transform
posterior = transform.(t, eachcol(pool_posterior_matrices(results)));
ps_t = posterior

l = @layout [a ; b; c]
pl_p = plot(getindex.(ps_t,:p),label="p"); hline!([θ0.p],label="")
pl_q = plot(getindex.(ps_t,:q),label="q"); hline!([θ0.q],label="")
pl_r = plot(getindex.(ps_t,:r),label="r"); hline!([θ0.r],label="")
plot(pl_p, pl_q, pl_r, layout=l)



# Extract the parameter.
posterior_p = first.(posterior);

# check the mean
mean(posterior_p)

# check the effective sample size
ess, R̂ = ess_rhat(stack_posterior_matrices(results))

# NUTS-specific statistics of the first chain
summarize_tree_statistics(results[1].tree_statistics)













# inplace version
∇negloglik_repam!(Πroot, ys) = (θ, storage) -> ForwardDiff.gradient!(storage, negloglik_repam(Πroot, ys), θ)
optimize(∇negloglik_repam(Πroot, ys), someθ)
optimize(∇negloglik_repam(Πroot, ys), ∇negloglik_repam!(Πroot, ys), someθ, Newton())


# ∇negloglik(Πroot, ys)(opt.minimizer)

# # inplace versions
# ∇negloglik!(Πroot, ys) = (θ, storage) -> ForwardDiff.gradient!(storage, negloglik(Πroot, ys), θ)


# storage = ∇negloglik(Πroot, ys)(someθ)
# ∇negloglik!(Πroot, ys)(someθ, storage)

optimize(negloglik(Πroot, ys), ∇negloglik!(Πroot, ys), someθ, Newton())
optimize(negloglik(Πroot, ys), someθ)
