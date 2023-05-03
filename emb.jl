import Pkg

# https://spcman.github.io/getting-to-know-julia/nlp/word-embeddings/
# https://nlp.stanford.edu/projects/glove/

# Pkg.add("Distances")
# Pkg.add("MultivariateStats")
# Pkg.add("WordTokenizers")
# Pkg.add("DelimitedFiles")
# Pkg.add("TextAnalysis")
Pkg.add("PyPlot")

using Distances, Statistics
using MultivariateStats
using PyPlot
using WordTokenizers
using TextAnalysis
using DelimitedFiles

function load_embeddings(embedding_file)
    local LL, indexed_words, index
    indexed_words = Vector{String}()
    LL = Vector{Vector{Float32}}()
    open(embedding_file) do f
        index = 1
        for line in eachline(f)
            xs = split(line)
            word = xs[1]
            push!(indexed_words, word)
            push!(LL, parse.(Float32, xs[2:end]))
            index += 1
        end
    end
    return reduce(hcat, LL), indexed_words
end

embeddings, vocab = load_embeddings("glove.42B.300d.txt")
vec_size, vocab_size = size(embeddings)
println("Loaded embeddings, each word is represented by a vector with $vec_size features. The vocab size is $vocab_size")

vec_idx(s) = findfirst(x -> x==s, vocab)
# vec_idx("cheese")

function vec(s) 
    if vec_idx(s)!=nothing
        embeddings[:, vec_idx(s)]
    end    
end
# vec("cheese")

cosine(x,y)=1-cosine_dist(x, y)
# cosine(vec("dog"), vec("puppy")) > cosine(vec("trousers"),vec("octopus"))

function closest(v, n=20)
    list=[(x,cosine(embeddings'[x,:], v)) for x in 1:size(embeddings)[2]]
    topn_idx=sort(list, by = x -> x[2], rev=true)[1:n]
    return [vocab[a] for (a,_) in topn_idx]
end

# closest(vec("water") + vec("frozen"))
# closest(mean([vec("day"), vec("night")]))