using Plots, Random, MatrixDepot, LinearAlgebra

#From Question_2.jl :

function arnoldi_method(A,b,m)
    n = size(b)[1]
    Q = zeros(n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(m+1,m)
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 
    return w,lambda
end

function hw1_mv_double_gs(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    g = Qk'*w
    w = w -Qk*g
    h = h + g
    beta = norm(w)
    return h, beta, w
end

#Function

# Compute the Rayleigh-Ritz method for K_m an unorthogonal matrix.
#   Input
# A : Matrix approximated
# b : Initial vector
# m : Size of K_m
#   Output
# w : eigenvectors
# lambda : eigenvalues
function rayleigh_ritz_method_km(A, b, m)
    K_m= Matrix{Float64}(undef, length(b), m)
    K_m[:,1] = b
    for i in 2:m
        K_m[:,i] = A * K_m[:,i-1]
    end
    lambda, z = eigen(inv(K_m'*K_m)*(K_m'*(A*K_m)))
    w = K_m*z
    return w, lambda
end

# Plot eigenvalues from both Rayleigh-Ritz method and from arnoldi method until M.
# nn : size of wathen matrix
# M : number of iterations 
function plot_compare_rayleigh_ritz_arnoldi(nn,M)
    Random.seed!(0);
    A = MatrixDepot.wathen(nn, nn)
    b = randn(size(A)[1])
    p = plot(xlabel="m", ylabel="Real part of eigenval. approx.")
    for m in 1:M 
        w1,lambda1 = rayleigh_ritz_method_km(A,b,m)
        w2,lambda2 = arnoldi_method(A,b,m)
        scatter!(p,fill(m,m) ,real.(lambda1) , color= :blue, label = (m==1 ? "Rayleigh-Ritz" : ""), markershape=:xcross) 
        scatter!(p,fill(m,m), real.(lambda2),color =:red, label =(m==1 ? "Arnoldi" : ""), markersize=3)   
    end
    p
end

# Main

plot_compare_rayleigh_ritz_arnoldi(30,11)