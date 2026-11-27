using LinearAlgebra, SparseArrays, MatrixDepot, PyPlot

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

    errors = zeros(m)
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
        z= (H[:,1:k]\e1 )* norm_b
        x = Q[:,1:k]*z 
        errors[k] = norm(A*x-b)/norm(b)
    end
    

    return errors
end 

function CG(A,b,m)  
    errors = zeros(m)
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
        errors[k] = norm(A*x-b)/norm(b)
    end
    return errors
end 
    
function CGLS(A,b,m)
    errors = zeros(m)
    X = zeros(size(b,1),m)
    x= zeros(size(b))
    r = b - A * x
    p = A' * r
    r_old = r
    for k = 1:m
        alpha = dot(p, p) / dot(A * p, A * p)
        x = x + alpha* p
        X[:,k] = x
        r = r_old  - alpha * (A * p)
        beta = dot(A' * r, A' * r) / dot(p, p)
        p = A' * r + beta*p
        r_old = r 
        B = A *X[:,1:k] #(n*n) *(n*m) = n*m
        z= inv(B'* B)*(B'*b) #(m*m)*(m*1) = m*1
        x= X[:,1:k]*z #(n*m)*(m*1) = n*1
        errors[k] = norm(A*x-b)/norm(b)
    end
    return errors
end



n = 100
e = ones(n)
A = -spdiagm(-1 => e[1:n-1], 0 => -6*e, 1 => e[1:n-1])# pas compris 
b = kron(ones(n÷2), [0.0, 1.0]) + ones(n)# pas compris 

m=30
errors_GMERS = GMERS(A,b,m)
errors_CG = CG(A,b,m)
errors_CGLS = CGLS(A,b,m)

figure(figsize=(12, 8))
semilogy(1:m, errors_CG, label="Method 1: CG", color="black", linestyle="", marker="*", markersize=8)
semilogy(1:m, errors_GMERS, label="Method 2: GMRES", color="blue", linestyle="", marker="o", markerfacecolor="none", markersize=7)
semilogy(1:m, errors_CGLS, label="Method 3: CG with LSQ", color="red", linestyle="", marker="+", markersize=8)
yticks([1e-17, 1e-12, 1e-7, 1e-2,1e3],["\$10^{-17}\$", "\$10^{-12}\$", "\$10^{-7}\$", "\$10^{-2}\$", "\$10^{3}\$"])  
ylim([1e-17, 1e3])
xticks([0, 5, 10, 15, 20, 25, 30],["0", "5", "10", "15", "20", "25", "30"])  
xlim([0, 30])
xlabel("Iteration \$p\$")
ylabel("\$||Ax-b||/||b||\$")
legend()
grid()
savefig("../Images/Question1/Q2A.png")
