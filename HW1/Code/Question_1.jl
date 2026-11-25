ENV["GKSwstype"] = "nul" #Remove the warning when plotting in a server without GUI
using LinearAlgebra, Plots



function find_two_closest_eigenvalue(A,mu)
    ev = eigvals(Symmetric(A))
    idx = sortperm(abs.(ev .- mu))      
    return ev[idx[1]], ev[idx[2]]
end


function power_method(A,v,n)
    
    ev= eigvals(Symmetric(A))
    lambda_1 =ev[end]
    lambda_2 = ev[end-1]
    
    err = zeros(n)
    rate_list = zeros(n)

    rate = lambda_2/lambda_1
    lambda =0.0

    w = similar(v)

    for k in 1:n
        w = A*v
        v = w/norm(w)
        lambda = v'*A*v
        err[k] = abs(lambda - lambda_1)
        rate_list[k] = abs((rate)^(2*(k-1)))
    end

    return err,rate_list
end


function inverse_iteration(A,v,mu,n)
    lambda_2, lambda_3 = find_two_closest_eigenvalue(A,mu)
    lambda = 0.0
    
    err = zeros(n)
    rate_list = zeros(n)
    rate = abs((lambda_2 - mu)/abs(lambda_3 - mu))

    w=similar(v)
    v = v/norm(v) 

    for k in 1:n
        w = inv((A - mu*I))*v
        v = w/norm(w)
        lambda = v'*A*v
        err[k] = abs(lambda - lambda_2)
        rate_list[k] = rate^(2*k)
    end

    return err,rate_list
end



function rayleigh_quotient_iteration(A,v,n)
    v = v/norm(v)
    lambda = v'*A*v                 
    lambdas = zeros(n)

    for k in 1:n
        w = inv((A - lambda*I))*v    
        v = w/norm(w)
        lambda = v'*A*v
        lambdas[k] = lambda
    end
    ev = eigvals(A)                
    lambda_star = ev[argmin(abs.(ev .- lambdas[end]))]
    err = abs.(lambdas .- lambda_star)
    return err
end

function plot_rayleigh_quotient_iteration(A,v,n)
    err = rayleigh_quotient_iteration(A, v, n)
    p= plot(1:n, err, yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
    savefig(p, "../Images/Question1/rayleigh_quotient_iteration_plot_bis.png")
end
function plot_inverse_iteration(A,v,mu,n)
    err,lambda_test = inverse_iteration(A, v, mu, n)
    err= max.(err, 1e-16)
    p= plot(1:n, [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
    savefig(p, "../Images/Question1/inverse_iteration_plot.png")
end
function plot_power_method(A,v,n)
    err,lambda_test = power_method(A, v, n)
    p= plot(1:n, [err,lambda_test], yscale=:log10, marker=[:o :none],color=[:blue :red],linestyle=[:solid :dash], xlabel="iteration", ylabel="err", legend=true,label=["observed" "theory"])
    savefig(p, "../Images/Question1/power_method_plot.png")
end


# Main

A = [4.0 1.0 1.0; 1.0 3.0 0.0; 1.0 0.0 2.0]
v = [1.0, 1.0, 1.0]
mu = 3.0
#plot_power_method(A,v,10)
#plot_inverse_iteration(A,v,mu,10)
#plot_rayleigh_quotient_iteration(A,v,3)

A = [4.0 1.0 1.0; 1.5 3.0 0.0; 1.0 0.0 2.0]

plot_rayleigh_quotient_iteration(A,v,4)