
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

