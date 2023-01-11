######### testing code for latentmarkov.jl ################
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
    
    @btime loglik(Πroot, 𝒪s)(θ0);           # 3.495 ms (59402 allocations: 4.47 MiB)
    @btime ∇loglik(Πroot, 𝒪s)(θ0);          # 13.773 ms (148972 allocations: 39.99 MiB)

    #---------------------- check type stability
    @code_warntype loglik(Πroot, 𝒪s[1])(θ0)
    @code_warntype loglik(θ0, Πroot, 𝒪s[1])
    @code_warntype loglik(θ0, Πroot, 𝒪s)

    using Cthulhu

    ####### ForwardDiff is faster and allocates less than FiniteDiff ###########
    TESTING = false
    if TESTING
        using FiniteDiff
        using BenchmarkTools
        ∇loglik_fd(Πroot, 𝒪) = (θ) -> FiniteDiff.finite_difference_gradient(loglik(Πroot, 𝒪), θ)
        @btime ∇loglik_fd(Πroot, 𝒪)(θ0);
        @btime ∇loglik(Πroot, 𝒪)(θ0);
    end




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

@code_warntype model.f(
    model,
    Turing.VarInfo(model),
    Turing.SamplingContext(
        Random.GLOBAL_RNG, Turing.SampleFromPrior(), Turing.DefaultContext(),
    ),
    model.args...,
)



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
