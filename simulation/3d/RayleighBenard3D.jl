# run: julia -> ] -> activate . -> backspace -> include("RayleighBenard3D.jl")

using Printf
using Oceananigans
using Statistics
using HDF5
using CUDA # when using GPU: (1) julia -> ] -> add CUDA (2) uncomment line 44


# script directory
dirpath = string(@__DIR__)

# domain size
L = (2*pi, 2*pi, 2) # x,y,z

# number of discrete sampled points
N = (48, 48, 32)

# time
Δt = 0.01 # simulation delta
Δt_snap = 0.3 # save delta
duration = 300 # duration of simulation

# temperature
min_b = 0 # Temperature at top plate
Δb = 1 # Temperature difference between bottom and top plate

# Rayleigh Benard Parameters
Ra = 10000
Pr = 0.71

# Set the amplitude of the random initial perturbation (kick)
random_kick = 0.2


function simulate_3d_rb(Ra=Ra, Pr=Pr, N=N, L=L, min_b=min_b, Δb=Δb, random_kick=random_kick,
    Δt=Δt, Δt_snap=Δt_snap, duration=duration)

    ν = sqrt(Pr * Δb * L[3]^3 / Ra)
    κ = ν / Pr

    grid = define_sample_grid(N, L)
    u_bcs, v_bcs, b_bcs = define_boundary_conditions(min_b, Δb)

    model = define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
    initialize_model(model, min_b, L[3], Δb, random_kick)

    totalsteps = Int(div(duration, Δt_snap))

    simulation_name = "$(N[1])_$(N[2])_$(N[3])_$(Ra)_$(Pr)_$(Δt)_$(Δt_snap)_$(duration)"
    h5_file, dataset, h5_file_path = create_hdf5_dataset(simulation_name, N, totalsteps)

    simulate_model(model, dataset, Δt, Δt_snap, totalsteps)

    close(h5_file)
    println("Simulation data saved as: $(h5_file_path)")
end

function define_sample_grid(N, L)
    # without GPU:
    # grid = RectilinearGrid(size=(N), x=(0, L[1]), y=(0, L[2]), z=(0, L[3]), 
    #  topology=(Periodic, Periodic, Bounded))
    # with GPU:
    grid = RectilinearGrid(GPU(), size=N, x=(0, L[1]), y=(0, L[2]), z=(0, L[3]),
        topology=(Periodic, Periodic, Bounded))
    return grid
end


function define_boundary_conditions(min_b, Δb)
    u_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    v_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(0),
        bottom=ValueBoundaryCondition(0))
    #! why are vertical velocities not bounded?
    # w_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(0),
    #                                 bottom = ValueBoundaryCondition(0))
    b_bcs = FieldBoundaryConditions(top=ValueBoundaryCondition(min_b),
        bottom=ValueBoundaryCondition(min_b + Δb))
    return u_bcs, v_bcs, b_bcs
end


function define_model(grid, ν, κ, u_bcs, v_bcs, b_bcs)
    model = NonhydrostaticModel(; grid,
        advection=UpwindBiasedFifthOrder(),
        timestepper=:RungeKutta3,
        tracers=(:b),
        buoyancy=Buoyancy(model=BuoyancyTracer()),
        closure=(ScalarDiffusivity(ν=ν, κ=κ)),
        boundary_conditions=(u=u_bcs, v=v_bcs, b=b_bcs,),
        coriolis=nothing
    )
    return model
end


function initialize_model(model, min_b, Lz, Δb, kick)
    # Set initial conditions
    uᵢ(x, y, z) = kick * randn()
    vᵢ(x, y, z) = kick * randn()
    wᵢ(x, y, z) = kick * randn()
    bᵢ(x, y, z) = min_b + (Lz - z) * Δb / 2 + kick * randn()

    # Send the initial conditions to the model to initialize the variables
    set!(model, u=uᵢ, v=vᵢ, w=wᵢ, b=bᵢ)
end


function create_hdf5_dataset(simulation_name, N, totalsteps)
    data_dir = joinpath(dirpath, "data", simulation_name)
    mkpath(data_dir) # create directory if not existent

    # compute number of this simulation
    i = 1
    while isfile(joinpath(data_dir, "sim$(i).h5"))
        i += 1
    end

    path = joinpath(data_dir, "sim$(i).h5")
    h5_file = h5open(path, "w")
    # save temperature and velocities in one dataset:
    dataset = create_dataset(h5_file, "data", datatype(Float64),
        dataspace(totalsteps + 1, 4, N...), chunk=(1, 1, N...))

    # seperate datasets for temperature and velocity:
    # temps = create_dataset(h5_file, "temperature", datatype(Float64),
    #     dataspace(totalsteps + 1, N...), chunk=(1, N...))
    # vels = create_dataset(h5_file, "velocity", datatype(Float64),
    #     dataspace(totalsteps + 1, 3, N...), chunk=(1, 1, N...))

    return h5_file, dataset, path
end


function simulate_model(model, dataset, Δt, Δt_snap, totalsteps)
    simulation = Simulation(model, Δt=Δt, stop_time=Δt_snap)
    simulation.verbose = true

    cur_time = 0.0

    # save initial state
    save_simulation_step(model, dataset, 1, N)

    for i in 1:totalsteps
        #update the simulation stop time for the next step
        global simulation.stop_time = Δt_snap * i

        run!(simulation)
        cur_time += Δt_snap

        save_simulation_step(model, dataset, i + 1, N)

        if (step_contains_NaNs(model, N))
            printstyled("[ERROR] NaN values found!\n"; color=:red)
            return
        end

        println(cur_time)
    end
end


function save_simulation_step(model, dataset, step, N)
    dataset[step, 1, :, :, :] = model.tracers.b[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 2, :, :, :] = model.velocities.u[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 3, :, :, :] = model.velocities.v[1:N[1], 1:N[2], 1:N[3]]
    dataset[step, 4, :, :, :] = model.velocities.w[1:N[1], 1:N[2], 1:N[3]]
end


function step_contains_NaNs(model, N)
    contains_nans = (any(isnan, model.tracers.b[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.u[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.v[1:N[1], 1:N[2], 1:N[3]]) ||
                     any(isnan, model.velocities.w[1:N[1], 1:N[2], 1:N[3]]))
    return contains_nans
end