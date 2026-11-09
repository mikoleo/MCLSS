using LinearAlgebra, Plots

#Eigenvalue method

# Compute the power method
#   Input : 
# A : Matrix studied
# v : Starting vector
# n : Number of iterations
#   Output :
# w : Eigenvector
# lambda : Eigenvalue
# err : Error between the eigenvalue approximated and the real eigenvalue
# lambda_1 : maximum of eigenvalues
# rate_list : Theorical error
function power_method(A,v,n)
    
    eigvalss = eigvals(Symmetric(A))
    lambda_1 = eigvalss[end]
    lambda_2 = eigvalss[end-1]
    
    err = zeros(n+1)
    rate_list = zeros(n+1)

    rate = lambda_2/lambda_1
    lambda =0.0

    w = similar(v)

    for k in 1:(n+1)
        w = A*v
        v = w/norm(w)
        lambda = transpose(v)*A*v
        err[k] = abs(lambda - lambda_1)
        rate_list[k] = abs((rate)^(2*(k-1)))
    end

    return w,lambda,err,lambda_1,rate_list
end


# Compute the inverse iteration
#   Input : 
# A : Matrix studied
# v : Starting vector
# mu : Starting eigvalue
# n : Number of iterations
#   Output :
# w : Eigenvector
# lambda : Eigenvalue
# err : Error between the eigenvalue approximated and the real eigenvalue
# lambda_2 : Eigevalue where the function converge if mu = 3.
# rate_list : Theorical error
function inverse_iteration(A,v,mu,n)
    lambda_2 = eigvals(Symmetric(A))[end-1]
    lambda_3 = eigvals(Symmetric(A))[end-2]
    lambda = 0.0

    err = zeros(n+1)
    rate_list = zeros(n+1)

    w = similar(v)
    v = w/norm(w)
    lambda = transpose(v)*A*v
    err[1] = abs(lambda - lambda_2)
    
    for k in 1:n
        w = inv((A - mu*I))*v
        v = w/norm(w)
        lambda = transpose(v)*A*v
        err[k+1] = abs(lambda - lambda_2)
    end

    for k in 0:n
        rate_list[k+1] = abs((lambda_2 - mu)/abs(lambda_3 - mu))^(2*k)
    end

    return w,lambda,err,lambda_2,rate_list
end

# Compute the rayleigh quotient iteration
#   Input : 
# A : Matrix studied
# v : Starting vector
# mu : Starting eigvalue
# n : Number of iterations
#   Output :
# w : Eigenvector
# lambda : Eigenvalue
# err : Error between the eigenvalue approximated and the real eigenvalue
# lambda_2 : Eigevalue where the function converge if mu = 3.
# rate_list : Theorical error
function rayleigh_quotient_iteration(A,v,mu,n)
    lambda_2 = eigvals(Symmetric(A))[end-1]
    lambda_3 = eigvals(Symmetric(A))[end-2]
    lambda = 0.0

    err = zeros(n+1)
    rate_list = zeros(n+1)

    w = similar(v)
    v = w/norm(w)
    lambda = mu
    err[1] = abs(lambda - lambda_2)
    
    for k in 1:n
        w = inv((A - mu*I))*v
        v = w/norm(w)
        lambda = transpose(v)*A*v
        err[k+1] = abs(lambda - lambda_2)
    end

    for k in 0:n
        rate_list[k+1] = abs((lambda_2 - mu)/abs(lambda_3 - mu))^(2*k)
    end

    return w,lambda,err,lambda_2,rate_list
end

# Plot

#Plot rayleigh quotient iteration
function plot_rayleigh_quotient_iteration(A,v,mu,n)
    w,lambda,err,lambda_1,lambda_test = rayleigh_quotient_iteration(A, v, mu, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end

#Plot inverse iteration
function plot_inverse_iteration(A,v,mu,n)
    w,lambda,err,lambda_1,lambda_test = inverse_iteration(A, v, mu, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end

#Plot power method
function plot_power_method(A,v,n)
    w,lambda,err,lambda_1,lambda_test = power_method(A, v, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end


# Main

A = [4.0 1.0 1.0; 1.5 3.0 0.0; 1.0 0.0 2.0]
v = [1.0, 1.0, 1.0]
n = 10
mu = 3.0
plot_power_method(A,v,n)