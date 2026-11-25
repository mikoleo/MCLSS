using Pkg, Plots, MAT, Statistics, LinearAlgebra

Pkg.add("MAT")

function init(faces)
    n = size(faces)[1]*size(faces)[2]
    m = size(faces)[3]
    X= Matrix{Int64}(undef, n, m)
    for i in 1:m
        X[:,i] = vec(faces[:,:,i])
    end
    return X
end




function questionC(Xc, n)
    error_method1 = zeros(n)
    error_method2 = zeros(n)
    m = size(Xc)[2]

    C = 1/(m-1) * (Xc * Xc')
    lambda_star = maximum(eigvals(C))
    println("Maximum eigenvalue of C: ", lambda_star)
    #method1 
    t= time()
    b = ones(size(C)[2])
    v = b/norm(b)
    for i in 1:n
        w = C*v
        v = w/norm(w)
        error_method1[i] = abs(lambda_star - (dot(v, w)))/abs(lambda_star)
    end
    time_taken = round(time() - t, digits=2)
    
    println("Time taken to compute power method with C for ", n, " iterations: ", time_taken, " seconds")
    println("Maximum eigenvalue computed using power method with C: ",v'*C*v," error: ", error_method1[n])
    p= plot(1:n, error_method1, yscale=:log10, marker=:o,color=:blue,linestyle=:solid, xlabel="iteration", ylabel="err", legend=true,label=["method with C"])
    eigenface = reshape(v, 64, 64)
    p_face = heatmap(eigenface; color = :grays, yflip = true, framestyle = :box)
    savefig(p_face, "../Images/Question7/eigenface_c_method_C.png")
    #methdod2
   
    b = ones(size(Xc')[2])
    v = b/norm(b)
    t= time()
    for i in 1:n
        w = (Xc*(Xc'*v))/(m-1)
        v = w/norm(w)
        error_method2[i] = abs(lambda_star - (dot(v, w)))/abs(lambda_star)
    end
    
    time_taken = round(time() - t, digits=2) 
    println("Time taken to compute power method with Xc for ", n, " iterations: ", time_taken, " seconds")
    println("Maximum eigenvalue computed using power method with Xc: ",(v'*Xc*Xc'*v)/(m-1)," error: ", error_method2[n])
    plot!(p, 1:n, error_method2, yscale=:log10, marker=:o,color=:red,label=["method with Xc"])
    savefig(p, "../Images/Question7/power_method.png")

    eigenface = reshape(v, 64, 64)
    p_face = heatmap(eigenface; color = :grays, yflip = true, framestyle = :box)
    savefig(p_face, "../Images/Question7/eigenface_c_method_Xc.png")
    return 
end

function powermethod(A, b, m)

    lambda_star = maximum(eigvals(A))
    error = zeros(m)
    v = b/norm(b)
    F =lu(A)
    L=F.L
    U=F.U
    for i in 1:m
        w = U\(L\v)
        v = w/norm(v)
        error[i] = abs(lambda_star - (v'*A*v))/abs(lambda_star)
    end
    lambda = v'*A*v
    return v, lambda
end

function hw1_gs_mv_single(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    beta = norm(w)
    return h, beta, w
end


function questionD(Xc, m)
#Pas fini 

    m1 = size(Xc)[2]
    error = zeros(m)

    #Arnoldi 
    b = ones(size(Xc')[2])
    n = size(b)[1]
    Q = zeros(Complex,n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(Complex,m+1,m)
    for k=1:m
        w = Xc*(Xc'*Q[:,k])/(m1-1)
        h, beta, worth = hw1_gs_mv_single(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 

    #Five 

end 

#A)
# p = heatmap(faces[:, :, 394]; color = :grays, yflip = true, framestyle = :box)
# savefig(p, "../Images/Question7/face394.png")

#B)
X = init(faces)

X_mean = mean(X, dims=2)
Xc = X .- X_mean
C = 1/(size(Xc)[2]-1) * (Xc * Xc')

# p =heatmap(reshape(X_mean, 64, 64),color=:grays, yflip=true, framestyle=:box)
# savefig(p, "../Images/Question7/mean_face.png")

# println("Size of Xc matrix: ", size(Xc))
# println("Size of C matrix: ", size(C))

# t = time()
# Xc * Xc' 
# time_taken = round(time() - t, digits=2)
# println("Time taken to compute C: ", time_taken, " seconds")

# m =100

# t= time()
# for i in 1:m
#     C*randn(size(C)[2])
# end 
# time_taken = round(time() - t, digits=2)
# println("Time taken to compute C*b for ", m, " random vectors: ", time_taken, " seconds")

# m =100

# t= time()
# for i in 1:m
#     Xc'*(Xc*randn(size(Xc)[2]))
# end 
# time_taken = round(time() - t, digits=2)
# println("Time taken to compute Xc'*(Xc*b) for ", m, " random vectors: ", time_taken, " seconds")

#C)
#questionC(Xc, 30)

#D)

questionD(Xc, 30)
