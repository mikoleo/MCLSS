
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
    n = size(b)[1]
    norm_b = norm(b)
    Q = zeros(n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(m+1,m)
    e1 = zeros(m+1)
    e1[1] = 1
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    z= (H[:,1:k]\e1 )* norm_b
    x = Q[:,1:k]*z 

    return x
end 

function CG(A,b,m)  
    x = zeros(size(b))
    r = b - A * x
    p = r
    r_old = r
    for k = 1:m
        alpha = dot(r, r) / dot(p, A * p)
        x = x + alpha* p
        r = r_old  - alpha * (A * p)
        beta = dot(r, r) / dot(r_old, r_old)
        p = r + beta*p
        r_old = r 
    end
    return x
end 


n = 100
e = ones(n)
A = -spdiagm(-1 => e[1:n-1], 0 => -6*e, 1 => e[1:n-1])# pas compris 
b = kron(ones(n÷2), [0.0, 1.0]) + ones(n)# pas compris 

