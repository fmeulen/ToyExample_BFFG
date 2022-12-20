wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using ComponentArrays


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
    hs = [hprev]
    loglik = zero(θ[1])
    for i=N:-1:1
        h = (K * hprev) .* Λ[:,ys[i]]  
        c = normalise!(h)
        loglik += c
        pushfirst!(hs, h)
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
N = 100

Ki(θ) = [1.0 - 0.5θ.p 0.25θ.p   0.25θ.p  ;
         0   1.0 - θ.p θ.p  
         0.4θ   0.3θ.p   1.0 - 0.7θ.p]

Λi(θ) = [1.0 0.0; 1.0 0.0; 0.0 1.0]
Πroot = [1.0, 1.0, 1.0]/3.0

# True parameter vector
θ0 = ComponentArray(p=0.3)
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
θgrid = [ComponentArray(p=x) for x in grid]
pl_lik = plot(grid, negloglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
# out = optimize(negloglik(Πroot, ys), 0.0, 1.0)    # box constrained optimization
# vline!(pl_lik, [out.minimizer], label="mle")
 vline!(pl_lik, [θ0], label="true")

someθ = θgrid[20]
negloglik(Πroot, ys)(someθ)
∇negloglik(Πroot, ys)(someθ)



