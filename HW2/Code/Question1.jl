using LinearAlgebra, SparseArrays, Random, MatrixDepot, PyPlot, Statistics

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
    x_star= A\b 
    relative_residual_error=zeros(m)
    relative_error = zeros(m)
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta

        z= (H[:,1:k]\e1 )* norm_b
        x = Q[:,1:k]*z 

        relative_residual_error[k] = norm((A*x)-b)/norm_b
        relative_error[k] = norm(x_star - x)/ norm(x_star)
    end
    return relative_error, relative_residual_error

end

function GMERS_Modified(A,b,m)
    n = size(b)[1]
    norm_b = norm(b)
    Q = zeros(n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(m+1,m)
    e1 = zeros(m+1)
    e1[1] = 1
    t = time()
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    z= (H\e1)* norm_b
    x= Q[:,1:m]*z
    t = time() - t
    residual_error = norm((A*x)-b)
    
    return t, residual_error

end

function computeDiskOutlier(eigenvalues,alpha)
    #Use the one outlier method 
    if alpha ==100 
        outlier_index = argmin(abs.(eigenvalues .- 0.9))
        outlier = eigenvalues[outlier_index]
        eigenvalues = deleteat!(eigenvalues, outlier_index)
        center =  mean(eigenvalues)
        radius = maximum(abs.(eigenvalues .- center))

    else
        outlier_index = argmin(abs.(eigenvalues .- 0.8))
        outlier = eigenvalues[outlier_index]
        eigenvalues = deleteat!(eigenvalues, outlier_index)
        center =  mean(eigenvalues)
        radius = maximum(abs.(eigenvalues .- center))
    end
    
    return outlier,center, radius
end

function computeTheoreticalConvergenceRateOutlier(A,center,radius,outlier,m)
    eigenvalues = eigvals(Matrix(A))
    F = eigen(Matrix(A))     
    V= F.vectors   
    rate = zeros(m)
    for k=1:m
        rate[k] = ((radius/abs(center))^k)*(((abs(outlier-center)+radius)*abs(center))/abs(outlier*radius))
    end
    rate = rate.*(norm(V)*norm(inv(V)))
    return rate
end

# n = 100
# A0 = sprand(n, n, 0.5)      
# b = rand(n)
# m=100
# figure(figsize=(12, 8))
# for (i,alpha) in enumerate([1, 5, 10, 100])
    
#     A = A0 + alpha * I
#     A = A / opnorm(A,1)
#     relative_error, relative_residual_error = GMERS(A,b,m)
    
#     eigenvalues = eigvals(Matrix(A))
#     outlier, center, radius = computeDiskOutlier(eigenvalues,alpha)
#     #Compute therietical convergence rate
#     rate = computeTheoreticalConvergenceRateOutlier(A,center,radius,outlier,m)
    

#     #Plot
#     subplot(2, 2, i)  
#     semilogy(1:m, relative_residual_error, "r", label="\$||Ax-b||/||b||\$")
#     semilogy(1:m, relative_error,color="black", linestyle="--", label="\$||x_{★}-x||/||x_{★}||\$")
#     semilogy(1:m, rate, color="blue", linestyle="--", label="Theory prediction")
#     yticks([1e-17, 1e-11, 1e-5, 1e1],["\$10^{-17}\$", "\$10^{-11}\$", "\$10^{-5}\$", "\$10^{1}\$"])  
#     ylim([1e-17, 1e1])
#     xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],["0", "10", "20", "30", "40", "50", "60", "70", "80", "90", "100"])  
#     xlim([0, 100])
#     xlabel("Iteration")
#     ylabel("Error")
#     title("\$\\alpha = $(alpha)\$")
#     legend()
#     grid()   
# end
# tight_layout()
# savefig("../Images/Question1/GMERS_all.png")


# #Question 1.2
# #Plot eigen values for each alpha
# figure(figsize=(12, 8))
# theta = range(0, 2*pi, length=100)
# for (i,alpha) in enumerate([1, 5, 10, 100])
#     A = A0 + alpha * I
#     A = A / opnorm(A,1)
#         eigenvalues = eigvals(Matrix(A))
    
#     outlier, center, radius = computeDiskOutlier(eigenvalues,alpha)
#     circle = [center + radius * exp(im * theta) for theta in theta]
#     subplot(2, 2, i)
#     scatter(real(eigenvalues), imag(eigenvalues), color="blue", marker="x", label="Eigvals")
#     yticks([-0.2, -0.1, 0, 0.1, 0.2],["\$-0.2\$", "\$-0.1\$", "\$0\$", "\$0.1\$", "\$0.2\$"])
#     xticks([-0.2, 0,0.2, 0.4, 0.6, 0.8, 1],["\$-0.2\$", "\$0\$", "\$0.2\$", "\$0.4\$", "\$0.6\$", "\$0.8\$", "\$1\$"])
#     xlim([-0.2, 1])
#     ylim([-0.2, 0.2])
#     xlabel("Re \$\\lambda\$")
#     ylabel("Im \$\\lambda\$")
#     title("\$\\alpha = $(alpha)\$")
#     plot(real(circle), imag(circle), color="red", label=" Localization Disk")
#     scatter(real(outlier), imag(outlier), color="green",label="Outlier", marker="x")
#     scatter(real(center), imag(center), color="orange",marker="o", label="Center")
#     legend()
#     grid()
# end
# tight_layout()
# savefig("../Images/Question1/Eigenvalues_all.png")

#Question 1.3

Alpha =[1,100]
M= [5,10,20,50,100]
N=[200,500,1000]

#Faire la compilation avant pour eviter erreurs de temps d'execution
n0 = 50
alpha0 = 10
m0= 5
a0 = sprand(n0, n0, 0.5)      
b0= rand(n0)
a1 = a0 + alpha0 * I
a1 = a1 / opnorm(a1,1)
_,_= GMERS_Modified(a1,b0,m0)
a1\b0



for n in N 
    A0 = sprand(n, n, 0.5)      
    b = rand(n)
    for alpha in Alpha
        A = A0 + alpha * I
        A = A / opnorm(A,1)
        for m in M 
            Time, Residual_error = 0, 0
            for i in 1:10
            t, residual_error = GMERS_Modified(A,b,m)
            Time += t
            Residual_error += residual_error
            end
            Time /= 10
            Residual_error /= 10
            println("alpha: ", alpha, " n: ", n, " m: ", m, " time: ", Time, " residual_error: ", Residual_error)  
        end
        println()
    end
    println()
    println()
end
println("Backslash operator")
for n in N 
    A0 = sprand(n, n, 0.5)      
    b = rand(n)
    for alpha in Alpha 
        A = A0 + alpha * I
        A = A / opnorm(A,1)
        Time = 0
        Residual_error = 0
        for i in 1:10
        t = time()
        x= A\b
        t = time() - t
        Time += t
        residual_error = norm((A*x)-b)
        Residual_error += residual_error
        end
        Time /= 10
        Residual_error /= 10
        println("alpha: ", alpha, " n: ", n," time: ", Time, " residual_error: ", Residual_error)
    
    end
    println()
    println()
end


