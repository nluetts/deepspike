using Pkg
Pkg.activate(@__DIR__)

using DelimitedFiles
using Flux
using Flux: train!
using GLMakie
using Statistics

function main()

    @info "load training data"

    mat = let
        spec = [readdlm("train/train_spec_$num", ',')[:,2] for num in 1:100]
        reduce(hcat, spec) .|> Float32
    end

    mask = repeat(vcat(repeat([true], 1340), repeat([false], 1340)), 6)
    nmask = repeat(vcat(repeat([false], 1340), repeat([true], 1340)), 6)

    @info "preprocess training data"

    X = mat[nmask, :]
    Y = mat[mask, :]
    target = Float32.(X .!= Y)

    loss(model, x, y) = mean(abs2.(sigmoid(model(x)) .- y));

    predict = Dense(8040 => 8040)
    opt = Flux.setup(RAdam(), predict)

    y = Observable([(0.0, 0.0)])

    fig, _, _ = lines(X[1:1340, 1])
    scatter!(y; color=:red)
    display(fig)

    @info "start training"

    for _ in 1:1000
    train!(loss, predict, [(X, target)], opt)
    @show loss(predict, X, target)
    y[] = filter(p -> p[2] > 0.1 && p[1] < 1340, collect(enumerate(sigmoid(predict(X[:, 1])) .* X[:, 1])))
    end

    predict
end
