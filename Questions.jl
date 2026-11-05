using LinearAlgebra, Plots

function power_method(A,v,n)
    
    eigvalss = eigvals(Symmetric(A))
    lambda_1 = eigvalss[end]
    lambda_2 = eigvalss[end-1]
    
    err = zeros(n+1)
    rate_list = zeros(n+1)

    rate = lambda_2/lambda_1
    

    w = similar(v)

    v = w/norm(w)
    lambda = transpose(v)*A*v
    err[1] = abs(lambda - lambda_1)
    for k in 1:n
        w = A*v
        v = w/norm(w)
        lambda = transpose(v)*A*v
        err[k+1] = abs(lambda - lambda_1)
    end

    for k in 0:n
        rate_list[k+1] = abs((rate)^(2*k))
    end

    return w,lambda,err,lambda_1,rate_list
end

function plot_power_method(A,v,n)
    w,lambda,err,lambda_1,lambda_test = power_method(A, v, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end

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


function plot_inverse_iteration(A,v,mu,n)
    w,lambda,err,lambda_1,lambda_test = inverse_iteration(A, v, mu, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end


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
        rate_list[k+1] = abs((lambda_2 - mu)/abs(lambda_3 - mu))^(k)
    end

    return w,lambda,err,lambda_2,rate_list
end

function plot_rayleigh_quotient_iteration(A,v,mu,n)
    w,lambda,err,lambda_1,lambda_test = rayleigh_quotient_iteration(A, v, mu, n)
    print(w,", ",lambda_1," and ",lambda)
    plot(1:(n+1), [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
end

plot_rayleigh_quotient_iteration([4 1 1;1.5 3 0;1 0 2], [1.0,1.0,1.0],3.0,10)
