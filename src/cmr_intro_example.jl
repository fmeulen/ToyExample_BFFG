wdir = @__DIR__
cd(wdir)
outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using QuadGK
using ComponentArrays
using DataFrames

include("funcdefs.jl")

EASYkernel = false
TESTING = true # if true: observe fully with known root node

# Need to specify
# Λ = observation kernel
# Πroot = prior on initial state
# Ki = transition kernel on hidden state



# Length of Markov chain
N = 50

# True parameter vector
# p0 =0.2; p1 = 0.7; q0= 0.5; q1=0.2
# θ0 = [p0, p1, q0, q1]
# κS = [1-p0 p0; p1 1-p1]  # high/low error transmission
# κB = [0.5 0.5 ; 0.5 0.5] # signal 
# κBS = kron(κB, κS)
# Λ = [1-q0 q0; 1-q1 q1; q0 1-q0; q1 1-q1];

#----------- Gilbert-Elliot model -------------------------------------------
# Define the state space (convention (B,S))
E = [(0,0), (0,1), (1,0), (1,1)]
V = [0, 1]

if TESTING    # if perfect observations, with known root node, we should be able to perfectly reconstruct 
     Πroot = [1.0, 0.0, 0.0, 0.0]
else 
     Πroot = ones(4)/4.0
end

θ0 = ComponentArray(p0=0.2, p1=0.7, q0= 0.01, q1=0.32)
Ki(θ) = kron( [0.5 0.5 ; 0.5 0.5], [1.0-θ.p0 θ.p0; θ.p1 1.0-θ.p1] )
Λi(θ) = [1.0-θ.q0 θ.q0; 1.0-θ.q1 θ.q1; θ.q0 1.0-θ.q0; θ.q1 1.0-θ.q1];


#----------- dishonest casino -------------------------------------------
Πroot = ones(2)/2.0
Πroot = [1.0, 0.0]
E = [1,2] # hidden states
V = collect(1:6) # observed states
θ0 = ComponentArray(p0=0.5, p1=0.9)
Ki(θ) = [θ.p0 1.0-θ.p0 ; 1.0-θ.p1 θ.p1]
Λi(θ) = hcat(ones(6)/6.0, [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]/10.0)'
Λi(θ) = hcat([5.0, 1.0, 1.0, 1.0, 1.0, 1.0]/10.0, [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]/10.0)'

############ ------------------------------------------------ ############

# generate track, and sample from conditioned process
N = 10_00
x, y = generate_track(Ki(θ0), Λi(θ0), Πroot, N)

# check
y1 = y[x.==1];  mean(y1.==6)
y2 = y[x.==2];  mean(y2.==6)

(ll, h) = loglik_and_bif(θ0, Πroot, y)
xᵒ = guided_track(Ki(θ0), Πroot, h)

out = DataFrame(latent=map(x -> E[x], x), observed=map(x -> V[x], y), inferred=map(x -> E[x], xᵒ))

ts = 0:N
#plot(ts, first.(out.latent), linestyle = :steps)
scatter(ts, first.(out.latent))
scatter!(ts, first.(out.inferred))

scatter(ts, first.(out.latent)- first.(out.inferred))

sum(out.latent.==out.inferred)/N


θgrid = [ComponentArray(p0=a, p1=θ0.p1) for a in collect(0:0.01:1)]
nll = negloglik(Πroot, y).(θgrid)
pl_lik = plot(first.(Θgrid), nll, label="log-likelihood")



# Monte Carlo
B = 100
Xᵒ = zeros(B, length(xᵒ))
for i in 1:B
     Xᵒ[i,:] =  guided_track(Ki(θ0), Πroot, h)
end

postprob1 = [mean(xᵒ.==2) for xᵒ in eachcol(Xᵒ)]
plot(ts, postprob1)
plot!(ts, x .- 1.0)










negloglik(Πroot, ys)(θ0)
∇negloglik(Πroot, ys)(θ0)

θinit = ComponentArray(p0=0.04, p1=0.3, q0= 0.51, q1=0.9)
negloglik(Πroot, ys)(θinit)
∇negloglik(Πroot, ys)(θinit)







loglikelihood(θ0, Πroot, ys)
loglikelihood2(θ0, Πroot, ys)



lower = fill(0.0, 4)
upper = fill(1.0, 4)

opt = optimize(negloglikelihood2(Πroot, ys), lower, upper, θinit)
opt.minimizer

opt = optimize(negloglik(Πroot, ys), θinit, LBFGS(); autodiff = :forward)
opt.minimizer
θ0

optimize(negloglik(Πroot, ys), ∇negloglik(Πroot, ys), θinit; inplace = false)

# plotting 
ts = 0:N
pl_paths = plot(ts, xstars, label="recovered")
plot!(pl_paths, ts, xs, label="latent", linestyle=:dash)
if TESTING
     plot!(pl_paths, ts, ys, label="observed")
else 
     plot!(pl_paths, ts, ys .+ 1, label="observed")
end
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
posterior(Λ, Πroot, ys) = (θ) -> likelihood(θ, Λ, Πroot, ys)*pdf(Πθ, θ)
Πprop = posterior(Λ, Πroot, ys)
Π(θ) = Πprop(θ) / quadgk(Πprop, 0, 1, rtol=1e-8)[1]

pl_post = plot(Θgrid, Π.(Θgrid), label="posterior")
vline!(pl_post, [out.minimizer], label="mle")
vline!(pl_post, [θ0], label="true")
pl_post

############ ------------------------------------------------ ############

# export files to pdf
savefig(pl_paths, joinpath(outdir,"paths.pdf"))
savefig(pl_lik, joinpath(outdir,"loglikelihood.pdf"))
savefig(pl_post, joinpath(outdir,"posterior_density.pdf"))