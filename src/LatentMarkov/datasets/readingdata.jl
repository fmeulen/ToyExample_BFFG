using CSV
using DataFrames
d = CSV.read("datasets/ToyExample.csv", DataFrame; delim=",")

TX = SVector{2,Float64}
TY = SVector{4,Int64}


n = 11
𝒪s = ObservationTrajectory{TX,TY}[]
for i ∈ 1:n
    di = d[d.subject .== i, :]
    X = TX[]
    Y = TY[]
    for r in eachrow(di)
        push!(X, SA[r[3], r[4]])
        push!(Y, SA[r[6]+1, r[7]+1, r[8]+1, r[9]+1])
    end
    push!(𝒪s, ObservationTrajectory(X,Y))
end    

