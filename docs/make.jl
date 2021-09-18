using StridedTensorXD
using Documenter

DocMeta.setdocmeta!(StridedTensorXD, :DocTestSetup, :(using StridedTensorXD); recursive=true)

makedocs(;
    modules=[StridedTensorXD],
    authors="PhysicsCodesLab",
    repo="https://github.com/PhysicsCodesLab/StridedTensorXD.jl/blob/{commit}{path}#{line}",
    sitename="StridedTensorXD.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://PhysicsCodesLab.github.io/StridedTensorXD.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/PhysicsCodesLab/StridedTensorXD.jl",
    devbranch="main",
)
