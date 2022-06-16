wdir = @__DIR__
cd(wdir)
#outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra
using Optim
using ForwardDiff
using Distributions
using QuadGK

function generate_track(K, Λ, proot, N)       # Generate exact track + observations
        xprev = sample(Weights(proot))  # root x_0
        xs = [xprev]
        ys = Int64[]# Vector{Int64}([])

        for i=1:N
                x = sample(Weights(K[xprev,:]))         # Generate sample from previous state
                push!(xs, x)
                y = sample(Weights(Λ[x,:]))             # Generate observation from sample
                push!(ys, y)
                xprev = x
        end
        (xs, ys)
end

function likelihood_and_guided(K, Λ, proot, ys, N)
        hprev = Λ[:,ys[N]]                   # Terminal h-transform is just pullback from leaf
        hs = [hprev]

        for i=N-1:-1:1
                h = K * hprev .* Λ[:,ys[i]]  # Pullback from leaf .* pullback from observation
                push!(hs, h)
                hprev = h
        end
        hroot = K* hprev
        hs = reverse(hs)                     # Reverse list for correctly indexed guiding terms
        likelihood = proot' * hroot          # likelihood = \int h_0(x_0) p(x_0) dx_0

        (likelihood, hroot, hs)
end

function guided_track(K, proot, hroot, hs, N)# Generate approximate track
        xprev = sample(Weights(proot .* hroot)) # Weighted prior distribution

        xstars = [xprev]

        for i=1:N
                p = K[xprev,:] .* hs[i]         # Weighted transition density
                x = sample(Weights(p))
                push!(xstars, x)
                xprev = x
        end
        xstars
end

# Define the state space
E = [1, 2, 3]

# Length of Markov chain
N = 28

# Transition kernel
Ki(θ) = [[1.0 - 0.5θ 0.25θ   0.25θ  ]
         [0   1.0 - θ θ  ]
         [0.4θ   0.3θ   1.0 - 0.7θ]]

Ki(θ) = [1.0 - θ  θ   0.0  ;
        0.25   0.5  0.25  ;
        0.4   0.3   0.3]



# Observation process
Λ = [[1. 0.]
     [1. 0.]
     [0. 1.]]

#      Λ = [[1. 0.]
#      [0.5 0.5]
#      [0. 1.]]


# Exact parameter vector
θ0 = 0.3

# Exact transition kernel
K = Ki(θ0)

# Root distribution
#proot = [1.0, 0.0, 0.0]
proot = [0.5, 0.25, 0.25]

xs, ys = generate_track(K, Λ, proot, N)
l, hroot, hs = likelihood_and_guided(K, Λ, proot, ys, N)
xstars = guided_track(K, proot, hroot, hs, N)

ts = 0:N
Q = plot(ts, xstars, label="recovered")
plot!(Q, ts, xs, label="latent", linestyle=:dash)

plot!(Q, 1:N, ys.+1, label="observed")


function likelihood(θ, Λ, proot, ys, N)
        K = Ki(θ)
        hprev = Λ[:,ys[N]]                   # Terminal h-transform is just pullback from leaf

        for i=N-1:-1:1
                h = K * hprev .* Λ[:,ys[i]]  # Pullback from leaf .* pullback from observation
                hprev = h
        end
        hroot = K* hprev
        likelihood = proot' * hroot          # likelihood = \int h_0(x_0) p(x_0) dx_0

        likelihood
end

llikelihood(θ) = log(likelihood(θ, Λ, proot, ys, N))



Θ = 0:0.01:1
lls = map(θ -> llikelihood(θ), Θ)
P = plot(Θ, lls, label="log-likelihood")

# box constrained optimization
negloglik(Λ, proot, ys, N) = (θ) ->  -log(likelihood(θ, Λ, proot, ys, N))
#negloglik(Λ, proot, ys, N) = (θ) ->  -log(likelihood_and_guided(Ki(θ), Λ, proot, ys, N)[1])
∇negloglik(Λ, proot, ys, N) = (θ) -> ForwardDiff.derivative(negloglik(Λ, proot, ys, N), θ)


out = optimize(negloglik(Λ, proot, ys, N), 0.0, 1.0)
print(out)
vline!(P, [out.minimizer], label="mle")
vline!(P, [θ0], label="true")

# check that derivative is zero at optimum (only works with first defn of negloklik). Why??
∇negloglik(Λ, proot, ys, N)(out.minimizer)


posterior(Λ, proot, ys, N) = (θ) ->  likelihood(θ, Λ, proot, ys, N)*pdf(Beta(1.0, 1.0), θ)
posterior(Λ, proot, ys, N)(0.2)



Πprop = posterior(Λ, proot, ys, N)
Π(θ) = Πprop(θ) / quadgk(Πprop, 0, 1, rtol=1e-8)[1]


Θgrid = collect(0:0.01:1)
plot(Θgrid, Π.(Θgrid), label="posterior")
vline!([θ0], label="true")