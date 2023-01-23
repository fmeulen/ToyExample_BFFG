using DataFrames

########### An example, where data are generated from the model ####################

# True parameter vector
γup = 2.0; γdown = -0.5
γ12 = γ23 = [γup, 0.0]
γ21 = γ32 = [γdown, -0.1]
Z0 = [0.5, 1.0, 1.5]
θ0 = ComponentArray(γ12 = γ12, γ21 = γ21, γ23 = γ23, γ32 = γ32, Z1=Z0, Z2=Z0, Z3=Z0, Z4=Z0)

println("true vals", "  ", γup,"  ", γdown,"  ", Z0)

# generate covariates, el1 = intensity, el2 = gender
n = 20 # nr of subjects
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
                push!(X, SA[-0.05*t + 0.2*randn(), 0.0])
            end
        else
            for t in 1: T
                push!(X, SA[-0.05*t + 0.2*randn(), 1.0])
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
            for t in 1: T
                push!(X, SA[-0.05*t + 0.2*randn(), 0.0])
            end
        else
            for t in 1: T
                push!(X, SA[-0.05*t + 0.2*randn(), 1.0])
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

lmestF <- lmestFormula(data=dout, response=5:8, LatentInitial=NULL, LatentTransition=3:4,AddInterceptInitial = FALSE,AddInterceptTransition = FALSE)

 
out0 = lmest(responsesFormula= lmestF$responsesFormula,
             latentFormula = lmestF$latentFormula,   
                index = c("subject", "time"),
                data = dt,
                k = 3,
                start = 0, # 0 deterministic, 1 random type of starting values
                modBasic = 1,
                seed = 123,
                tol = 1e-2) 

out1 <- lmest(responsesFormula = y1 + y2 + y3 + y4 ~ NULL,
              latentFormula = ~ 1 | x1 + x2,
              index = c("subject", "time"),
              data = dt,
              k = 3,
              start = 0, # 0 deterministic, 1 random type of starting values
              modBasic = 1,
              seed = 123,
              tol = 1e-2,)
summary(out1)
# lambdas = out1$Psi
# gammas = out1$Ga
"""

lmest_fit0 = @rget out0
lmest_fit1 = @rget out1

lmest_fit0[:Ga]

model = logtarget_large(𝒪s);

# compute map and mle 
@time map_estimate = optimize(model, MAP())

@time mle_estimate = optimize(model, MLE())

θmap =  map_estimate.values
θmle =  mle_estimate.values

mapZtoλ(θ0.Z1) 
mapallZtoλ(θmap) 
mapallZtoλ(θmle)

lmest_fit[:Psi] #lambdas

sampler =  NUTS(1000, 0.65) 
@time chain = sample(model, sampler, MCMCDistributed(), 1000, 4)#; progress=true);

# plotting 
histogram(chain)
plot(chain)
θpostmean = describe(chain)[1].nt.mean
mapallZtoλ(θpostmean)
