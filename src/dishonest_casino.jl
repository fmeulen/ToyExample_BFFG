# use functions from testprogram.jl

Πroot = ones(2)/2.0

E = [1,2] # hidden states
V = collect(1:6) # observed states
θ0 = ComponentArray(p0=0.95, p1=0.15)    # stay in honest state with prob p0, switch back from dishonest to honest state with prob 1-p1
Ki(θ) = [θ.p0 1.0-θ.p0 ; 1.0-θ.p1 θ.p1]
Λi(θ) = hcat(ones(6)/6.0, [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]/10.0)'
#Λi(θ) = hcat([5.0, 1.0, 1.0, 1.0, 1.0, 1.0]/10.0, [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]/10.0)'

# generate track, and sample from conditioned process
K = Ki(θ0)
Λ = Λi(θ0)
N = 1000
xs, ys = generate_track(K, Λ, Πroot, N)
logprior(θ)  = 0.0
(l, hs) = loglik_and_bif(θ0, Πroot, ys)
xstars = guided_track(K, Πroot, hs)

ts = 0:N
pl_paths = plot(ts, xstars, label="recovered")
plot!(pl_paths, ts, xs, label="latent", linestyle=:dash)
#plot!(pl_paths, ts, ys , label="observed")
pl_paths

# compute mle and plot with loglikelihood
grid = 0:0.01:1
θgrid = [ComponentArray(p0=x, p1=θ0.p1) for x in grid]
pl_lik = plot(grid, loglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
 vline!(pl_lik, [θ0.p0], label="true")


p = loglik(Πroot, ys)

t = as((p0=as_unit_interval , p1=as_unit_interval))
P = TransformedLogDensity(t, p)
@assert P.log_density_function(θ0) == p(θ0)

∇P = ADgradient(:ForwardDiff, P);


# one chain
outhmc = mcmc_with_warmup(Random.default_rng(2), ∇P, 1000)
ps = outhmc.posterior_matrix

ps_t = transform.(t, eachcol(ps))

l = @layout [a  b;  c d ]
pl_p = plot(getindex.(ps_t,:p0),label="p0"); hline!([θ0.p0],label="")
pl_p2 = histogram(getindex.(ps_t,:p0),label=""); vline!([θ0.p0],label="")
pl_q = plot(getindex.(ps_t,:p1),label="p1"); hline!([θ0.p1],label="")
pl_q2 = histogram(getindex.(ps_t,:p1),label=""); vline!([θ0.p1],label="")

plot(pl_p, pl_p2, pl_q, pl_q2, layout=l)


############ making it more difficult ###################
# Now we don't know exactly how the unfair dice look like by introducing the parameter c

θ0 = ComponentArray(p0=0.95, p1=0.15, c =5.0)    # stay in honest state with prob p0, switch back from dishonest to honest state with prob 1-p1
Ki(θ) = [θ.p0 1.0-θ.p0 ; 1.0-θ.p1 θ.p1]
Λi(θ) = hcat(ones(6)/6.0, [1.0, 1.0, 1.0, 1.0, 1.0, θ.c]/(5.0 + θ.c))'


# generate track, and sample from conditioned process
K = Ki(θ0)
Λ = Λi(θ0)
N = 10000
xs, ys = generate_track(K, Λ, Πroot, N)
logprior(θ)  = logpdf(Normal(5.0, 3.0), θ.c) + logpdf(Beta(1.0, 4.0), θ.p1)
(l, hs) = loglik_and_bif(θ0, Πroot, ys)
xstars = guided_track(K, Πroot, hs)

ts = 0:N
pl_paths = plot(ts, xstars, label="recovered")
plot!(pl_paths, ts, xs, label="latent", linestyle=:dash)
#plot!(pl_paths, ts, ys , label="observed")
pl_paths

# compute mle and plot with loglikelihood
grid = 0:0.01:1
θgrid = [ComponentArray(p0=x, p1=θ0.p1, c = θ0.c) for x in grid]
pl_lik = plot(grid, loglik(Πroot, ys).(θgrid), label="neg. loglikelihood")
 vline!(pl_lik, [θ0.p0], label="true")


p = loglik(Πroot, ys)

t = as((p0=as_unit_interval , p1=as_unit_interval, c = asℝ₊ ))
P = TransformedLogDensity(t, p)
@assert P.log_density_function(θ0) == p(θ0)

∇P = ADgradient(:ForwardDiff, P);


# one chain
outhmc = mcmc_with_warmup(Random.default_rng(2), ∇P, 1000)
ps = outhmc.posterior_matrix

ps_t = transform.(t, eachcol(ps))

l = @layout [a  b;  c d; e f ]
pl_p = plot(getindex.(ps_t,:p0),label="p0"); hline!([θ0.p0],label="")
pl_p2 = histogram(getindex.(ps_t,:p0),label=""); vline!([θ0.p0],label="")
pl_q = plot(getindex.(ps_t,:p1),label="p1"); hline!([θ0.p1],label="")
pl_q2 = histogram(getindex.(ps_t,:p1),label=""); vline!([θ0.p1],label="")
pl_r = plot(getindex.(ps_t,:c),label="c"); hline!([θ0.c],label="")
pl_r2 = histogram(getindex.(ps_t,:c),label=""); vline!([θ0.c],label="")
plot(pl_p, pl_p2, pl_q, pl_q2, pl_r, pl_r2, layout=l)


