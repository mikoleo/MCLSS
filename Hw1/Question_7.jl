using Pkg, Plots, MAT, Statistics

Pkg.add("MAT")

function init(faces)
    n = size(faces)[1]*size(faces)[2]
    m = size(faces)[3]
    println(n)
    println(m)
    X= Matrix{Int64}(undef, n, m)
    for i in 1:m
        X[:,i] = vec(faces[:,:,i])
    end
    return X
end

data = matread("Hw1/olivetti_faces.mat")

faces = data["faces"]

heatmap(faces[:, :, 394]; color = :grays, yflip = true, framestyle = :box)


X = init(faces)
