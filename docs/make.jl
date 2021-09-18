using Documenter
using StridedTensorXD

makedocs(modules=[StridedTensorXD],
            sitename="StridedTensorXD.jl",
            authors = "PhysicsCodesLab",
            format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true",
                                        mathengine = MathJax()),
            pages = [
                "Home" => "index.md",
                "Index" => ["index/index.md"]
            ])

deploydocs(repo = "github.com/PhysicsCodesLab/StridedTensorXD.jl.git")
