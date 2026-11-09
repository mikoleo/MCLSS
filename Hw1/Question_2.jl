using MatrixDepot, Random, LinearAlgebra

# Gram Schmidt

# Compute Gram Schmidt orthogonalization with the single for version.
#   Input
# Q : Orthogonal matrix
# w : Orthogonal complement
# k : Number of iterations
#   Output
# h : Gram-schmidt coefficient
# beta : Norm of orthogonal complement
# w : New orthogonal complement
function hw1_gs_single(Q,w,k)
    h = zeros(k)
    worth = copy(w)
    for i=1:k
        h[i] = Q[:,i]'*w
        worth = worth - h[i]*Q[:,i]
    end
    beta = norm(worth)
    return h, beta, worth
end

# Compute Gram Schmidt orthogonalization with the modfied for version.
#   Input
# Q : Orthogonal matrix
# w : Orthogonal complement
# k : Number of iterations
#   Output
# h : Gram-schmidt coefficient
# beta : Norm of orthogonal complement
# w : New orthogonal complement
function hw1_gs_modified(Q,w,k)
    h = zeros(k)
    worth = copy(w)
    for i=1:k
        h[i] = Q[:,i]'*worth
        worth = worth - h[i]*Q[:,i]
    end
    beta = norm(worth)
    return h, beta, worth
end

# Compute Gram Schmidt orthogonalization with the single MV version.
#   Input
# Q : Orthogonal matrix
# w : Orthogonal complement
# k : Number of iterations
#   Output
# h : Gram-schmidt coefficient
# beta : Norm of orthogonal complement
# w : New orthogonal complement
function hw1_gs_mv_single(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    beta = norm(w)
    return h, beta, w
end

# Compute Gram Schmidt orthogonalization with the double MV version.
#   Input
# Q : Orthogonal matrix
# w : Orthogonal complement
# k : Number of iterations
#   Output
# h : Gram-schmidt coefficient
# beta : Norm of orthogonal complement
# w : New orthogonal complement
function hw1_gs_mv_double(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    g = Qk'*w
    w = w -Qk*g
    h = h + g
    beta = norm(w)
    return h, beta, w
end

# Compute Gram Schmidt orthogonalization with the triple MV version.
#   Input 
# Q : Orthogonal matrix
# w : Orthogonal complement
# k : Number of iterations
#   Output
# h : Gram-schmidt coefficient
# beta : Norm of orthogonal complement
# w : New orthogonal complement
function hw1_gs_mv_triple(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    g = Qk'*w
    w = w -Qk*g
    f = Qk'*w
    w = w - Qk*f
    h = h + g + f
    beta = norm(w)
    return h, beta, w
end

# Arnoldi

# Compute the arnoldi method
#   Input
# A : Matrix approximated
# b : Initial vector
# m : Size of Q_m
#   Output 
# w : eigenvectors
# lambda : eigenvalues
function arnoldi_method(A,b,m)
    n = size(b)[1]
    Q = zeros(n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(m+1,m)
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_single_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 
    return w,lambda
end


# Print Arnoldi method for different size m
# nn : size of the wathen matrix
function plot_arnoldi_method(nn)
    Random.seed!(0);
    A = MatrixDepot.wathen(nn, nn) 
    b = randn(size(A)[1])
    for m in [5,10,20,50,100]
        t0 = time()
        w,lambda = arnoldi_method(A,b,m)
        t1 = time()
        println("m : ",m)
        println("orth : ",norm(Q'*Q - I))
        println("time : ",t1-t0)
    end
end

# Main

plot_arnoldi_method(1500)