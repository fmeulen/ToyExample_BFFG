

########### An example, where data are generated from the model ####################

# True parameter vector
γup = [2.0, 0.0]
γdown = [-0.5, -0.5]
Z1 = [0.5, 1.0, 1.5]
Z2 = [0.5, 1.0, 1.5]
Z3 = [0.2, 1.0, 2.5]
Z4 = [0.5, 1.0, 1.5]
θ0 = ComponentArray(γ12 = γup, γ21 = γdown, γ23 = γup, γ32 = γdown, Z1=Z1, Z2=Z2, Z3=Z3, Z4=Z4)

println("true vals", "  ", γup,"  ", γdown,"  ", Z1, Z2, Z3, Z4)

# generate covariates, el1 = intensity, el2 = gender
n = 25 # nr of subjects
T = 50 # nr of times at which we observe


INCLUDE_MISSING  = false

if INCLUDE_MISSING
    TX = Union{Missing, SVector{DIM_COVARIATES,Float64}} # indien er missing vals zijn 
    TY = Union{Missing, SVector{DIM_RESPONSE, Int64}}
  
    𝒪s = ObservationTrajectory{TX, TY}[]
    Us =  Vector{Int64}[]
    for i in 1:n
        #local X 
        X = TX[]   # next, we can push! elements to X
        if i ≤ 10 
            for t in 1: T
                push!(X, SA[-0.05*t + 0.02*randn(), 0.0])
            end
        else
            for t in 1: T
                push!(X, SA[-0.05*t + 0.02*randn(), 1.0])
            end
            X[3] = missing
        end
        U, Y =  sample(θ0, X) 
        push!(Us, U)
        YY = TY[]
        push!(YY, missing) 
        for t in  2:T
            push!(YY, Y[t]) 
        end    
        push!(𝒪s, ObservationTrajectory(X, YY))
    end
else 
    TX = SVector{2,Float64}
    TY = SVector{DIM_RESPONSE, Int64}

    𝒪s = ObservationTrajectory{TX, TY}[]
    Us =  Vector{Int64}[]
    for i in 1:n
        #local X 
        X = TX[]   # next, we can push! elements to X
        if i ≤ 10 
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 0.0])
            end
        else
            slope = rand(Uniform(-0.05,0.05))
            for t in 1: T
                push!(X, SA[slope*t + 0.1*randn(), 1.0])
            end
        end
        U, Y =  sample(θ0, X) 
        push!(Us, U)
        YY = TY[]
        for t in  1:T
            push!(YY, Y[t]) 
        end    
        push!(𝒪s, ObservationTrajectory(X, YY))
    end
end

#### convert the simulated data to a Julia-dataframe
out = []
for i ∈ 1:n
    𝒪 = 𝒪s[i]
    @unpack X, Y = 𝒪
    Y = [Y[j] .- SA[1,1,1,1] for j in eachindex(Y)]
    xx=vcat(X'...)
    yy=vcat(Y'...)
    ni = size(yy)[1]
    push!(out, hcat(fill(i,ni),1:ni,xx,yy))
end



dout = DataFrame(vcat(out...), :auto)
colnames = ["subject", "time", "x1", "x2", "y1", "y2", "y3", "y4"]
rename!(dout, colnames)

#CSV.write("testdatalatentmarkov.csv", dout)

#### Fit with LMest #####
using RCall
@rput dout
R"""
library(LMest)

#require(LMest)
dt <- lmestData(data = dout, id = "subject", time="time")

lmestF <- lmestFormula(data=dout, response=5:8, 
                        LatentInitial=NULL, 
                        LatentTransition=3:4,
                        AddInterceptInitial = FALSE,
                        AddInterceptTransition = FALSE)

 
out0 = lmest(responsesFormula= lmestF$responsesFormula,
             latentFormula = lmestF$latentFormula,   
                index = c("subject", "time"),
                data = dt,
                k = 3,
                start = 0, # 0 deterministic, 1 random type of starting values
                modBasic = 1,
                seed = 123,
                tol = 1e-2) 

# out1 <- lmest(responsesFormula = y1 + y2 + y3 + y4 ~ NULL,
#               latentFormula = ~ 1 | x1 + x2,
#               index = c("subject", "time"),
#               data = dt,
#               k = 3,
#               start = 0, # 0 deterministic, 1 random type of starting values
#               modBasic = 1,
#               seed = 123,
#               tol = 1e-2,)
# summary(out1)
# lambdas = out1$Psi
# gammas = out1$Ga
"""

lmest_fit0 = @rget out0
#lmest_fit1 = @rget out1

lmest_fit0[:Ga]

lmest_fit0[:Piv]

@show lmest_fit0[:Psi] # bottom row should resemble are lambdas
# get the bottom rows, the following should be close (if estimates are good)
@show [lmest_fit0[:Psi][:,:,1][2,:],lmest_fit0[:Psi][:,:,2][2,:],lmest_fit0[:Psi][:,:,3][2,:],lmest_fit0[:Psi][:,:,4][2,:]]
@show mapallZtoλ(θ0)'


#################### Fitting with Turing.jl ##########################

model = logtarget(𝒪s);
model = logtarget_large(𝒪s);

#--------------- map -----------------------
@time map_estimate = optimize(model, MAP());
θmap = convert_turingoutput(map_estimate);
@show mapallZtoλ(θ0)'
@show mapallZtoλ(θmap)'

@show θ0[:γ12], θmap[:γ12]
@show θ0[:γ21], θmap[:γ21]

#--------------- mle -----------------------
@time mle_estimate = optimize(model, MLE())
θmle = convert_turingoutput(mle_estimate);
@show mapallZtoλ(θ0)'
@show mapallZtoλ(θmle)'

@show θ0[:γ12], θmle[:γ12]
@show θ0[:γ21], θmle[:γ21]

#--------------- NUTS sampler -----------------------

sampler =  NUTS() 
@time chain = sample(model, sampler, MCMCDistributed(), 1000, 3)#; progress=true);

# plotting 
histogram(chain)
plot(chain)

# extract posterior mean
θpm = describe(chain)[1].nt.mean
θpm = ComponentArray(γ12=θpm[1:2], γ21=θpm[3:4], Z1=θpm[5:7], Z2=θpm[8:10],Z3=θpm[11:13],Z4=θpm[14:16])

@show mapallZtoλ(θpm)'
@show mapallZtoλ(θ0)'

@show θ0[:γ12], θpm[:γ12]
@show θ0[:γ21], θpm[:γ21]

Z1symb=[Symbol("Z1[1]"), Symbol("Z1[2]"), Symbol("Z1[3]")]
plot(chain[Z1symb])
savefig("Z1s.pdf")

γsymb=[Symbol("γup[1]"), Symbol("γup[2]"), Symbol("γdown[1]"), Symbol("γdown[2]")]
plot(chain[γsymb])
savefig("gammas.pdf")