wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using QuadGK

include("funcdefs.jl")

EASYkernel = false
TESTING = false # if true: observe fully with known root node

# Need to specify
# Λ = observation kernel
# Πroot = prior on initial state
# Ki = transition kernel on hidden state


# Define the state space
E = [1, 2, 3]

# Length of Markov chain
N = 400

if EASYkernel
        Ki(θ) = [[1.0 - 0.5θ 0.25θ   0.25θ  ]
         [0   1.0 - θ θ  ]
         [0.4θ   0.3θ   1.0 - 0.7θ]]
else
        Ki(θ) = [1.0 - θ  θ   0.0  ;
        0.25   0.5  0.25  ;
        0.4   0.3   0.3]
end

if TESTING    # if perfect observations, with known root node, we should be able to perfectly reconstruct 
     Λ = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
     Πroot = [1.0, 0.0, 0.0]
else 
     Λ = [1.0 0.0; 1.0 0.0; 0.0 1.0]
     Πroot = [0.5, 0.25, 0.25]
end

# True parameter vector
θ0 = 0.3

# Hidden Markov chain ransition kernel
K = Ki(θ0)

############ ------------------------------------------------ ############

# generate track, and sample from conditioned process
xs, ys = generate_track(K, Λ, Πroot, N)
l, hs = likelihood_and_guided(K, Λ, Πroot, ys)
xstars = guided_track(K, Πroot, hs)
# plotting 
ts = 0:N
pl_paths = plot(ts, xstars, label="recovered")
plot!(pl_paths, ts, xs, label="latent", linestyle=:dash)

plot!(pl_paths, ts, ys, label="observed")
pl_paths


############ ------------------------------------------------ ############

# compute mle and plot with loglikelihood
Θgrid = collect(0:0.01:1)
pl_lik = plot(Θgrid, negloglik(Λ, Πroot, ys).(Θgrid), label="log-likelihood")
out = optimize(negloglik(Λ, Πroot, ys), 0.0, 1.0)    # box constrained optimization
print(out)
vline!(pl_lik, [out.minimizer], label="mle")
vline!(pl_lik, [θ0], label="true")
pl_lik
# check that derivative is zero at optimum (only works with first defn of negloklik). Why??
∇negloglik(Λ, Πroot, ys)(out.minimizer)

############ ------------------------------------------------ ############

# draw posterior density 
Πθ = Beta(1.0, 1.0)
posterior(Λ, Πroot, ys) = (θ) ->  likelihood(θ, Λ, Πroot, ys)*pdf(Πθ, θ)
Πprop = posterior(Λ, Πroot, ys)
Π(θ) = Πprop(θ) / quadgk(Πprop, 0, 1, rtol=1e-8)[1]

pl_post = plot(Θgrid, Π.(Θgrid), label="posterior")
vline!(pl_post, [θ0], label="true")
pl_post

############ ------------------------------------------------ ############

# export files to pdf
savefig(pl_paths, joinpath(outdir,"paths.pdf"))
savefig(pl_lik, joinpath(outdir,"loglikelihood.pdf"))
savefig(pl_post, joinpath(outdir,"posterior_density.pdf"))