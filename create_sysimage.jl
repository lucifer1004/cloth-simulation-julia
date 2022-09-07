import Pkg

Pkg.activate(".")
Pkg.instantiate()

using PackageCompiler

create_sysimage(["Makie", "GLMakie", "Meshes", "MeshViz"], sysimage_path="sys_makie.so", precompile_execution_file="mass_spring_3d.jl")
