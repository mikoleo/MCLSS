using MAT, LinearAlgebra, SparseArrays
Pkg.add("MAT")


vars = matread("HW2/Matlab/cgn_illustration.mat")

A = vars["A"]
b = vars["b"]


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

function GMERS(A,b,m)
    T = promote_type(eltype(A),eltype(b))
    errors = zeros(real(T),m)
    n = size(b)[1]
    norm_b = norm(b)
    Q = zeros(T,n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(T,m+1,m)
    e1 = zeros(T,m+1)
    e1[1] = 1
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
        z= (H[:,1:k]\e1 )* norm_b
        x = Q[:,1:k]*z 
        errors[k] = norm(A*x-b)/norm(b)
    end
    return errors
end 

function CGN(A,b,m) 
    errors = zeros(m)
    x = zeros(size(b))
    r = A' *b 
    p = copy(r)
    
    r_old = r
    for k = 1:m
        AtAp = A'*A*p    

        alpha = dot(r, r) / dot(p, AtAp)
        x = x + alpha* p
        r = r_old  - alpha * AtAp
        beta = dot(r, r) / dot(r_old, r_old)
        p = r + beta*p
        r_old = r 
        errors[k] = norm(A*x-b)/norm(b)
    end
    return errors
end 




error = CGN(A,b,10)
print(error)
