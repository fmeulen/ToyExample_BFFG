wdir = @__DIR__
cd(wdir)
#outdir= joinpath(wdir, "out")

using StatsBase, Plots, LinearAlgebra

# Define the state space
E = [1, 2, 3]

# Length of Markov chain
N = 100

# Transition kernel
Ki(θ) = [[1-θ[1] θ[1]   0     ]
         [0      1-θ[2] θ[2]  ]
         [θ[3]   0      1-θ[3]]]

# Observation process
Λ = [[1. 0.]
     [1. 0.]
     [0. 1.]]

#Λ = Matrix(1.0*I(3))

# Exact parameter vector
θ = (0.4, 0.3, 0.2)

# Exact transition kernel
K = Ki(θ)

# Root state
xroot = 1

# Generate exact track + observations
xprev = xroot        # root x_0
xs = [xprev]     # root -1 is observed
ys = Vector{Int64}([])

for i=1:N
        x = sample(Weights(K[xprev,:]))         # Generate sample from previous state
        push!(xs, x)
        y = sample(Weights(Λ[x,:]))             # Generate observation from sample
        push!(ys, y)
        xprev = x
end

# Terminal h-transform is just pullback from leaf
hprev = Λ[:,ys[N]]
hs = [hprev]

for i=N-1:-1:1
        h = K * hprev .* Λ[:,ys[i]]             # Pullback from leaf .* pullback from observation
        push!(hs, h)
        hprev = h
end

# Reverse list for correctly indexed guiding terms
hs = reverse(hs)

# Generate approximate track
xprev = xroot
xstars = [xprev]

for i=1:N
        p = K[xprev,:] .* hs[i]                 # Weighted transition density
        x = sample(Weights(p))
        push!(xstars, x)
        xprev = x
end

ts = 0:N
plot(ts, xstars, label="recovered")
plot!(ts, xs, label="latent", linestyle=:dash)

plot!(1:N, ys.+1.0, label="observed")
