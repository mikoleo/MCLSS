using MatrixDepot, Random, LinearAlgebra

function arnoldi(A,b,m)
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
    return Q,H
end

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

function hw1_mv_single_gs(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    beta = norm(w)
    return h, beta, w
end

function hw1_mv_double_gs(Q,w,k)
    Qk = Q[:,k]
    h= Qk'*w
    w = w - Qk*h 
    g = Qk'*w
    w = w -Qk*g
    h = h + g
    beta = norm(w)
    return h, beta, w
end

function hw1_mv_triple_gs(Q,w,k)
    Qk = Q[:,k]
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


function plot_arnoldi_method(nn)
    Random.seed!(0);
    A = MatrixDepot.wathen(nn, nn) 
    b = randn(size(A)[1])
    for m in [5,10,20,50,100]
        t0 = time()
        Q,H = arnoldi(A,b,m)
        t1 = time()
        println("m : ",m)
        println("orth : ",norm(Q'*Q - I))
        println("time : ",t1-t0)
    end
end
plot_arnoldi_method(1500)