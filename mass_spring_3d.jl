using LinearAlgebra
using Meshes: Vec3, Point3, Ball, Ngon, SimpleMesh, connect
using MeshViz: viz, viz!
using Makie: Makie, ispressed, cam3d!, meshscatter!, center!, Scene, Mouse, Keyboard, Observable, Rect, @lift
using GLMakie: GLMakie
using ColorTypes: RGB

let
    n = 128
    quad_size = 1.0 / n
    dt = 4e-2 / n
    substeps = Int(ceil(1 / 60 / dt))
    gravity = Vec3(0, 0, -9.8)
    spring_Y = 3e4
    dashpot_damping = 1e4
    drag_damping = 1

    ball_radius = 0.3
    ball_center = Point3(0, 0, 0)

    x = [Point3(0.0, 0, 0) for _ in 1:n, _ in 1:n]
    v = zeros(Vec3, n, n)
    num_triangles = (n - 1) * (n - 1) * 2
    indices = fill((0, 0, 0), num_triangles)
    vertices = [Point3(0.0, 0, 0) for _ in 1:n*n]

    bending_strings = false

    function initialize_mass_points()
        rx = (rand() - 0.5) * 0.1
        ry = (rand() - 0.5) * 0.1

        for i in 1:n, j in 1:n
            x[i, j] = Vec3(
                i * quad_size - 0.5 + rx,
                j * quad_size - 0.5 + ry,
                0.6,
            )
        end
    end

    initialize_mass_points()

    function initialize_mesh_indices()
        for i in 1:n-1, j in 1:n-1
            k = (i - 1) * (n - 1) + j
            indices[2*k-1] = ((i - 1) * n + j, (i - 1) * n + j + 1, i * n + j)
            indices[2*k] = (i * n + j, (i - 1) * n + j + 1, i * n + j + 1)
        end
    end

    initialize_mesh_indices()

    spring_offsets = []
    if bending_strings
        for i in -1:1, j in -1:1
            if (i, j) != (0, 0)
                push!(spring_offsets, CartesianIndex(i, j))
            end
        end
    else
        for i in -2:2, j in -2:2
            if (i, j) != (0, 0) && abs(i) + abs(j) <= 2
                push!(spring_offsets, CartesianIndex(i, j))
            end
        end
    end

    function substep()
        Threads.@threads for idx in CartesianIndices(x)
            v[idx] = v[idx] + gravity * dt
        end

        Threads.@threads for i in CartesianIndices(x)
            force = Vec3(0, 0, 0)
            for offset in spring_offsets
                j = i + offset
                if 1 <= j[1] <= n && 1 <= j[2] <= n
                    x_ij = x[i] - x[j]
                    v_ij = v[i] - v[j]
                    current_dist = norm(x_ij)
                    d = x_ij / current_dist
                    original_dist = quad_size * norm(Tuple(offset))
                    force = force - spring_Y * d * (current_dist / original_dist - 1) - v_ij ⋅ d * d * dashpot_damping * quad_size
                end
            end

            v[i] = v[i] + force * dt
        end

        Threads.@threads for i in CartesianIndices(x)
            v[i] = v[i] * exp(-drag_damping * dt)
            offset_to_center = x[i] - ball_center
            if norm(offset_to_center) <= ball_radius
                normal = offset_to_center / norm(offset_to_center)
                v[i] = v[i] - min(v[i] ⋅ normal, 0) * normal
            end
            x[i] = x[i] + dt * v[i]
        end
    end

    function update_vertices()
        for i in 1:n, j in 1:n
            vertices[(i-1)*n+j] = x[i, j]
        end
    end

    GLMakie.activate!()
    GLMakie.set_window_config!(;
        vsync=false,
        framerate=60.0,
        float=false,
        pause_rendering=false,
        focus_on_show=false,
        decorated=true,
        title="Julia Cloth Simulation on GLMakie (Press Q to exit)"
    )
    scene = Scene(
        backgroundcolor=:white,
        ssao=Makie.SSAO(),
        lights=Makie.automatic,
        resolution=(500, 500),
    )
    glfw_window = GLMakie.to_native(display(scene))
    relative_space = Makie.camrelative(scene)

    # Draw the ball, use coefficient 0.95 to prevent visual penetration
    viz!(scene, Ball(ball_center, ball_radius * 0.95), color=:gray)
    plt = Ref{Any}(nothing)
    cam3d!(scene)
    center!(scene)

    running = true
    time = Observable(0.0)
    update_vertices()
    connec = connect.(indices, Ngon)
    mesh = SimpleMesh(vertices, connec)

    function update()
        while running
            if time[] > 1.5
                initialize_mass_points()
                time[] = 0.0
            end

            for _ in 1:substeps
                substep()
                time[] += dt
            end
            update_vertices()
            if !isnothing(plt[])
                delete!(scene, plt[])
            end
            plt[] = viz!(scene, mesh, color=1:length(indices))
        end

        GLMakie.destroy!(glfw_window)
    end

    b = @task update()
    schedule(b)

    GLMakie.on(scene.events.keyboardbutton) do event
        if event.key == Keyboard.q
            running = false
        end
    end

    wait(b)
end
