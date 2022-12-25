
"""
generate_track(K, Λ, Π, proot, N)

K: kernel on latent state (matrix)
Λ: kernel to produce observations
Πroot: prior on x0 (root node is like x_{-1}), represented by a column vector

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
            #pushfirst!(hs, h)
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




    
# function likelihood_and_guided(K, Λ, Πroot, ys)
#     N = length(ys) - 1
#     hprev = Λ[:,ys[N+1]]                   # Terminal h-transform is just pullback from leaf
#     hs = [hprev]

#     for i=N:-1:1
#             h = K * hprev .* Λ[:,ys[i]]  # Pullback from leaf .* pullback from observation
#             push!(hs, h)
#             hprev = h
#     end
#     # hroot = K* hprev # should not be done!!!
#     # hs = reverse(hs)                     # Reverse list for correctly indexed guiding terms
#     # likelihood = proot' * hroot          # likelihood = \int h_0(x_0) p(x_0) dx_0

#     #hroot = K* hprev
#     hs = reverse(hs)                     # Reverse list for correctly indexed guiding terms
#     likelihood = Πroot' * hprev          # likelihood = \int h_0(x_0) p(x_0) dx_0

#     (likelihood, hs)
# end

# function loglikelihood_wrong(θ, Πroot, ys)
# # wrong implementation, underflow risk
#     N = length(ys) - 1
#     K = Ki(θ)
#     Λ = Λi(θ)
#     hprev = Λ[:,ys[N+1]]     
#     for i=N:-1:1
#             hprev = (K * hprev) .* Λ[:,ys[i]]  
#             @show hprev
#     end
#     log(Πroot' * hprev)          
# end



# negloglikelihood2(Πroot, ys) = (θ) ->  -loglikelihood2(θ, Πroot, ys)

# function guided_track(K, Πroot, hs)# Generate approximate track
#     N = length(hs) - 1
#     xprev = sample(Weights(Πroot .* hs[1])) # Weighted prior distribution

#     xstars = [xprev]

#     for i=1:N
#             p = K[xprev,:] .* hs[i+1]         # Weighted transition density
#             x = sample(Weights(p))
#             push!(xstars, x)
#             xprev = x
#     end
#     xstars
# end


# negloglik(Πroot, ys) = (θ) ->  -log(likelihood(θ, Πroot, ys))
# #negloglik(Λ, Πroot, ys) = (θ) ->  -log(likelihood_and_guided(Ki(θ), Λ, Πroot, ys)[1])  # autodiff does not work on this one

# ∇negloglik(Πroot, ys) = (θ) -> ForwardDiff.gradient(negloglik(Πroot, ys), θ)