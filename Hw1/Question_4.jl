using MAT
using Plots
using LinearAlgebra
using Statistics

function hw1_gs_mv_single(Q,w,k)
    Qk = Q[:,1:k]
    h= Qk'*w
    w = w - Qk*h 
    beta = norm(w)
    return h, beta, w
end

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

function arnoldi_method_2(A,b,m)
    n = size(b)[1]
    Q = zeros(Complex,n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(Complex,m+1,m)
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_mv_double_gs(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 
    return w,lambda
end

#From the Question3 file

function arnoldi_method_1(A,b,m)
    n = size(b)[1]
    Q = zeros(Complex,n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(Complex,m+1,m)
    for k=1:m
        w = A*Q[:,k]
        h, beta, worth = hw1_gs_mv_single(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 
    return w,lambda
end

function shiftAndInvertAnorldiMethod(A,mu,b,m)
    n = size(b)[1]
    Q = zeros(Complex,n,m+1)
    Q[:,1] = b/norm(b)
    H= zeros(Complex,m+1,m)
    Id= 1*Matrix{Complex}(I,size(A)[1],size(A)[2])
    for k=1:m
        w = (A - mu*Id)\Q[:,k]
        h, beta, worth = hw1_gs_mv_single(Q,w,k)
        H[1:k,k] = h
        H[k+1,k] = beta
        Q[:,k+1] = worth/beta
    end
    lambda,z = eigen(H[1:m,1:m])
    w = Q[:,1:m]*z 
    return w,lambda
end

function NClosest(t_eigvals, compute_eigvals,n)
    eigvals= zeros(Complex,n)
    closest = zeros(Complex,size(compute_eigvals))
    err= zeros(size(compute_eigvals))
    for i in 1:size(compute_eigvals)[1]
        idx = argmin(abs.(t_eigvals .- compute_eigvals[i]))
        closest[i] = t_eigvals[idx]
        err[i] = abs(t_eigvals[idx] - compute_eigvals[i])
    end
        sort_idx = sortperm(err)
        eigvals= closest[sort_idx[1:size(eigvals)[1]]]
    return eigvals
end




function ComputeCenterAndCercle(t_eigvals, outlier)
    eigs = vec(copy(t_eigvals))   
    n = length(outlier)
    centerMass = zeros(ComplexF64, n)
    radius = zeros(Float64, n)

    for i in 1:n
        t = deleteat!(copy(outlier), i)
        j  = argmin(abs.(eigs .- outlier[i]))  
        t1 = deleteat!(copy(eigs), j)
        centerMass[i] = mean(t)
        radius[i] = maximum(abs.(t1 .- centerMass[i]))
    end

    println("Center of Mass: ", centerMass)
    println("Radius: ", radius)
    return centerMass, radius
end



B=matread("Bwedge.mat")["B"];
B_eigvals =matread("Bwedge.mat")["B_eigvals"];

#B

#p = scatter(real(B_eigvals), imag(B_eigvals), xlabel="Real Part", ylabel="Imaginary Part", legend=:outertopright, label="Eigenvalues")

#_, lambda = arnoldi_method_1(B, randn(size(B)[2]), 30)
#outlier = NClosest(B_eigvals, lambda, 3)

# scatter!(p, real(outlier), imag(outlier), color=:red, label="Outliers")

# centerMass, radius = ComputeCenterAndCercle(B_eigvals, outlier)
# theta = range(0, 2π, length=100)
# colors = [:orange, :green, :purple]

# for i in 1:length(outlier)
#     x = real(centerMass[i]) .+ radius[i] .* cos.(theta)
#     y = imag(centerMass[i]) .+ radius[i] .* sin.(theta)

#     plot!(p, x, y, color=colors[i], linewidth=2, label="Circle $i")
#     scatter!(p, [real(centerMass[i])], [imag(centerMass[i])],color=colors[i], markershape=:x, markersize=6,label="Center of Mass $i")
# end

# savefig(p, "../Images/Question4/b.png")


# ratiosConvergence =radius ./ abs.(outlier .- centerMass)

# println("Ratios of Convergence:")
# for i in 1:length(outlier)
#     color = colors[i]
#     println("Outlier related to Circle $i and Center of Mass $i (color $color): ", outlier[i], " Ratio: ", ratiosConvergence[i])
# end

#C 


# m=[2,4,8,10,20,30,40]
# markers = [:diamond, :xcross, :cross, :circle, :diamond, :circle, :circle]
# colors  = [:black, :orange, :green, :red, :brown, :peru, :deeppink]

# p = scatter(real(B_eigvals), imag(B_eigvals), xlabel="Real Part", ylabel="Imaginary Part",markershape=:circle,color=:blue, legend=:bottomleft, label="iteration 2")
# for i in 1:length(m)
#     _, lambda = arnoldi_method_2(B, randn(size(B)[2]), m[i])
#     scatter!(p, real(lambda), imag(lambda), markershape=markers[i], color=colors[i], label="iteration $(m[i])")
    
# end
# savefig(p, "../Images/Question4/c.png")

# I expect fast convergence to the eigvalues which seem to be the furthest from the center of mass. 

# Je ne sais pas comment m'y prendre pour calculer l'erreur avec le 10⁻¹⁰


#d 
lambda_target = -9.8 + 2im
real_lambda_target = B_eigvals[argmin(abs.(B_eigvals .- lambda_target))]
eigvals = zeros(ComplexF64,3,length(B_eigvals))
sigma_shift= [-10,-7+2im,-9.8+1.5im]
transformed_lambda_target =zeros(ComplexF64,3)
for i in 1:3
    eigvals[i,:] = 1 ./ (B_eigvals .- sigma_shift[i])
    transformed_lambda_target[i] = 1 / (real_lambda_target - sigma_shift[i])

end


# titres = ["sigma = -10", "sigma = -7 + 2i", "sigma = -9.8 + 1.5i"]

# for i in 1:3
#     p= scatter( real(eigvals[i,:]), imag(eigvals[i,:]), xlabel="Real Part", ylabel="Imaginary Part", label="transformed eigenvalues", color=:blue, markershape=:circle,legend=:outertopright,title=titres[i])
#     scatter!(p, [real(transformed_lambda_target[i])], [imag(transformed_lambda_target[i])], color=:red, markershape=:x, markersize=6, label="eigenvalue target")
#     savefig(p, "../Images/Question4/d$i.png")
# end


#e 

m=[10,20,30]
color = [:blue,:green,:orange]
for i in 1:length(sigma_shift)
    for j in 1:length(m)
        _, lambda3 = shiftAndInvertAnorldiMethod(B, sigma_shift[i], ones(size(B)[2]), m[j])
        lambda_orig = sigma_shift[i] .+ 1 ./ lambda3                    
        closest_idx = argmin(abs.(lambda_orig .- real_lambda_target))   
        err= abs(lambda_orig[closest_idx] - real_lambda_target)  
        closest_eig = lambda3[closest_idx]    
        println("For sigma = $(sigma_shift[i]) and m = $(m[j]) : error = $(err), closest eigenvalue = $(closest_eig)")   
    end
end



