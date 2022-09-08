using LinearAlgebra
using Meshes: Vec3f, Point3f, Ball, Ngon, SimpleMesh, connect
using MeshViz: viz, viz!
using GLMakie: GLMakie, Makie, ispressed, cam3d!, meshscatter!, center!, Scene, Mouse, Keyboard, Observable, Rect, VideoStream, recordframe!, save, @lift

function simulate(record=false)
    n = 128
    quad_size = 1.0f0 / n
    dt = 0.04f0 / n
    substeps = Int(ceil(1 / 60 / dt))
    gravity = Vec3f(0, 0, -9.8)
    spring_Y = 30000.0f0
    dashpot_damping = 10000.0f0
    drag_damping = 1

    balls = [
        (Point3f(0, 0, 0), 0.2f0),
        (Point3f(-0.5, 0, 0), 0.2f0),
        (Point3f(0.5, 0, 0), 0.2f0),
        (Point3f(0, 0.5, 0), 0.2f0),
        (Point3f(0, 0, -0.5), 0.2f0),
    ]
    ball_radius = 0.3f0
    ball_center = Point3f(0.0, 0, 0)

    x = [Point3f(0.0, 0, 0) for _ in 1:n, _ in 1:n]
    v = zeros(Vec3f, n, n)
    num_triangles = (n - 1) * (n - 1) * 2
    indices = fill((0.0f0, 0.0f0, 0.0f0), num_triangles)
    vertices = [Point3f(0.0, 0, 0) for _ in 1:n*n]

    bending_strings = false

    function initialize_mass_points()
        rx = (rand(Float32) - 0.5f0) * 0.1f0
        ry = (rand(Float32) - 0.5f0) * 0.1f0

        for i in 1:n, j in 1:n
            x[i, j] = Vec3f(
                i * quad_size - 0.5f0 + rx,
                j * quad_size - 0.5f0 + ry,
                0.6f0,
            )
        end
    end

    function initialize_mesh_indices()
        for i in 1:n-1, j in 1:n-1
            k = (i - 1) * (n - 1) + j
            indices[2*k-1] = ((i - 1) * n + j, (i - 1) * n + j + 1, i * n + j)
            indices[2*k] = (i * n + j, (i - 1) * n + j + 1, i * n + j + 1)
        end
    end

    spring_offsets = CartesianIndex{2}[]
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
        Threads.@threads for i in CartesianIndices(x)
            v[i] = v[i] + gravity * dt
        end

        Threads.@threads for i in CartesianIndices(x)
            force = Vec3f(0, 0, 0)
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
            for (ball_center, ball_radius) in balls
                offset_to_center = x[i] - ball_center
                if norm(offset_to_center) <= ball_radius
                    normal = offset_to_center / norm(offset_to_center)
                    v[i] = v[i] - min(v[i] ⋅ normal, 0.0f0) * normal
                end
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
        framerate=30,
        float=false,
        pause_rendering=false,
        focus_on_show=true,
        decorated=true,
        title="Julia Cloth Simulation on GLMakie (Press DEL to exit)"
    )
    scene = Scene(
        backgroundcolor=:white,
        ssao=GLMakie.SSAO(),
        lights=Makie.automatic,
        resolution=(1280, 720),
    )
    glfw_window = GLMakie.to_native(display(scene))

    running = true
    time = 0.0
    initialize_mesh_indices()
    initialize_mass_points()
    update_vertices()
    connec = connect.(indices, Ngon)
    mesh = SimpleMesh(vertices, connec)
    plt = Ref(viz!(scene, mesh, color=1:length(indices)))

    # Draw the ball, use coefficient 0.95 to prevent visual penetration
    for (ball_center, ball_radius) in balls
        viz!(scene, Ball(ball_center, ball_radius * 0.95f0), color=:gray)
    end
    cam3d!(scene)
    center!(scene)

    function update()
        if record
            stream = VideoStream(scene, framerate=24)
        end
        while running
            @show time
            if time > 1.5
                initialize_mass_points()
                time = 0.0
            end
            for _ in 1:substeps
                substep()
                time += dt
            end
            update_vertices()
            if !isnothing(plt[])
                delete!(scene, plt[])
            end
            plt[] = viz!(scene, mesh, color=1:length(indices))
            if record
                recordframe!(stream)
            end
        end

        if record
            save("cloth_simulation.mp4", stream)
        end
        GLMakie.destroy!(glfw_window)
    end

    GLMakie.on(scene.events.keyboardbutton) do event
        if event.key == Keyboard.delete
            running = false
        end
    end

    update()
end

simulate(true)
