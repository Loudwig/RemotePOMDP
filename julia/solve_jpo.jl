#!/usr/bin/env julia

using NativeSARSOP
using POMDPs
using POMDPTools

struct ArrayPOMDP <: POMDP{Int,Int,Int}
    transitions::Array{Float64,3} # state, next_state, action
    observations::Array{Float64,3} # observation, next_state, action
    rewards::Matrix{Float64} # state, action
    discount_factor::Float64
    initial_belief::Vector{Float64}
end

POMDPs.states(model::ArrayPOMDP) = 1:size(model.transitions, 1)
POMDPs.actions(model::ArrayPOMDP) = 1:size(model.transitions, 3)
POMDPs.observations(model::ArrayPOMDP) = 1:size(model.observations, 1)
POMDPs.stateindex(::ArrayPOMDP, state::Int) = state
POMDPs.actionindex(::ArrayPOMDP, action::Int) = action
POMDPs.obsindex(::ArrayPOMDP, observation::Int) = observation
POMDPs.discount(model::ArrayPOMDP) = model.discount_factor
POMDPs.initialstate(model::ArrayPOMDP) = SparseCat(
    collect(states(model)), model.initial_belief
)
POMDPs.isterminal(::ArrayPOMDP, ::Int) = false
POMDPs.transition(model::ArrayPOMDP, state::Int, action::Int) = SparseCat(
    collect(states(model)), collect(@view model.transitions[state, :, action])
)
POMDPs.observation(model::ArrayPOMDP, action::Int, next_state::Int) = SparseCat(
    collect(observations(model)),
    collect(@view model.observations[:, next_state, action]),
)
POMDPs.reward(model::ArrayPOMDP, state::Int, action::Int) = (
    model.rewards[state, action]
)

"""Native blind lower bound shifted into a certified Bellman subsolution."""
mutable struct CertifiedBlindLower <: Solver
    base::NativeSARSOP.BlindLowerBound
    max_raw_residual::Float64
    max_subsolution_shift::Float64
end

CertifiedBlindLower(; bel_res::Float64, max_time::Float64) = CertifiedBlindLower(
    NativeSARSOP.BlindLowerBound(bel_res=bel_res, max_time=max_time),
    0.0,
    0.0,
)

function POMDPs.solve(
    solver::CertifiedBlindLower,
    pomdp::NativeSARSOP.ModifiedSparseTabular,
)
    raw_policy = solve(solver.base, pomdp)
    corrected_alphas = Vector{Vector{Float64}}()
    policy_actions = Int[]
    gamma = discount(pomdp)
    solver.max_raw_residual = 0.0
    solver.max_subsolution_shift = 0.0

    for (raw_alpha, action) in alphapairs(raw_policy)
        alpha = collect(raw_alpha)
        bellman_alpha = (
            @view(pomdp.R[:, action])
            + gamma * transpose(pomdp.T[action]) * alpha
        )
        # If w = alpha-c*1 and c >= max(alpha-B(alpha))/(1-gamma),
        # then w <= B(w). Monotonicity makes w a certified lower bound on
        # the value of the feasible policy which repeats this action.
        shift = max(
            0.0,
            maximum(alpha .- bellman_alpha) / (1.0 - gamma),
        )
        solver.max_raw_residual = max(
            solver.max_raw_residual,
            maximum(abs.(alpha .- bellman_alpha)),
        )
        push!(corrected_alphas, alpha .- shift)
        push!(policy_actions, action)
        solver.max_subsolution_shift = max(
            solver.max_subsolution_shift,
            shift,
        )
    end
    return AlphaVectorPolicy(pomdp, corrected_alphas, policy_actions)
end

"""Upper bound obtained by revealing the physical state to the controller."""
mutable struct FullyObservableUpper <: Solver
    bel_res::Float64
    max_time::Float64
    residual::Float64
    iterations::Int
end

FullyObservableUpper(; bel_res::Float64, max_time::Float64) = (
    FullyObservableUpper(bel_res, max_time, Inf, 0)
)

function POMDPs.solve(
    solver::FullyObservableUpper,
    pomdp::NativeSARSOP.ModifiedSparseTabular,
)
    gamma = discount(pomdp)
    n_states = length(states(pomdp))
    value = fill(maximum(pomdp.R) / (1.0 - gamma), n_states)
    updated = similar(value)
    start_time = time()
    solver.residual = Inf
    solver.iterations = 0

    while true
        fill!(updated, -Inf)
        for action in actions(pomdp)
            action_value = (
                @view(pomdp.R[:, action])
                + gamma * transpose(pomdp.T[action]) * value
            )
            updated .= max.(updated, action_value)
        end
        solver.residual = maximum(abs.(updated .- value))
        solver.iterations += 1
        value, updated = updated, value
        if (
            solver.residual <= solver.bel_res
            || time() - start_time >= solver.max_time
        )
            break
        end
    end

    # V_MDP is linear in a belief and upper-bounds the partially observed
    # problem. The action label is unused when SARSOP extracts corner values.
    return AlphaVectorPolicy(pomdp, [value], [first(actions(pomdp))])
end

function read_metadata(path::String)
    metadata = Dict{String,String}()
    for line in eachline(path)
        isempty(strip(line)) && continue
        key, value = split(line, '\t'; limit=2)
        metadata[key] = value
    end
    return metadata
end

function read_float64(path::String, dimensions::Tuple)
    count = prod(dimensions)
    values = open(path, "r") do io
        buffer = Vector{Float64}(undef, count)
        read!(io, buffer)
        buffer
    end
    length(values) == count || error("wrong number of values in $path")
    return reshape(values, dimensions)
end

function write_float64(path::String, values)
    open(path, "w") do io
        write(io, Float64.(vec(values)))
    end
end

function write_int64(path::String, values)
    open(path, "w") do io
        write(io, Int64.(vec(values)))
    end
end

function solve_and_export(input_directory::String, output_directory::String)
    metadata = read_metadata(joinpath(input_directory, "metadata.tsv"))
    n_states = parse(Int, metadata["n_states"])
    n_physical_states = parse(Int, metadata["n_physical_states"])
    n_actions = parse(Int, metadata["n_actions"])
    n_observations = parse(Int, metadata["n_observations"])
    gamma = parse(Float64, metadata["gamma"])
    export_beliefs = get(metadata, "export_beliefs", "true") == "true"

    transitions = read_float64(
        joinpath(input_directory, "transitions.bin"),
        (n_states, n_states, n_actions),
    )
    observation_matrix = read_float64(
        joinpath(input_directory, "observations.bin"),
        (n_observations, n_states, n_actions),
    )
    rewards = read_float64(
        joinpath(input_directory, "rewards.bin"),
        (n_states, n_actions),
    )
    initial_belief = vec(read_float64(
        joinpath(input_directory, "initial_belief.bin"),
        (n_states,),
    ))

    model = ArrayPOMDP(
        transitions,
        observation_matrix,
        rewards,
        gamma,
        initial_belief,
    )
    initial_lower = CertifiedBlindLower(
        bel_res=parse(Float64, metadata["initial_bound_residual"]),
        max_time=parse(Float64, metadata["initial_bound_max_time"]),
    )
    initial_upper_method = metadata["initial_upper_bound"]
    initial_upper = if initial_upper_method == "fully_observable"
        FullyObservableUpper(
            bel_res=parse(Float64, metadata["initial_bound_residual"]),
            max_time=parse(Float64, metadata["initial_bound_max_time"]),
        )
    elseif initial_upper_method == "fib"
        NativeSARSOP.FastInformedBound(
            bel_res=parse(Float64, metadata["initial_bound_residual"]),
            max_time=parse(Float64, metadata["initial_bound_max_time"]),
            init_value=maximum(rewards) / (1.0 - gamma),
        )
    else
        error("unsupported initial upper bound: $initial_upper_method")
    end
    solver = NativeSARSOP.SARSOPSolver(
        epsilon=parse(Float64, metadata["search_epsilon"]),
        precision=parse(Float64, metadata["precision"]),
        kappa=parse(Float64, metadata["kappa"]),
        delta=parse(Float64, metadata["delta"]),
        max_time=parse(Float64, metadata["max_time"]),
        max_steps=parse(Int, metadata["max_steps"]),
        verbose=false,
        prunethresh=parse(Float64, metadata["prune_threshold"]),
        use_binning=metadata["use_binning"] == "true",
        init_lower=initial_lower,
        init_upper=initial_upper,
    )

    initialization_start = time()
    tree = NativeSARSOP.SARSOPTree(solver, model)
    initialization_seconds = time() - initialization_start
    initial_lower_residual = solver.init_lower.max_raw_residual
    initial_upper_residual = if solver.init_upper isa FullyObservableUpper
        solver.init_upper.residual
    else
        maximum(solver.init_upper.residuals)
    end
    initial_upper_iterations = if solver.init_upper isa FullyObservableUpper
        solver.init_upper.iterations
    else
        -1
    end
    initial_lower_subsolution_shift = solver.init_lower.max_subsolution_shift
    history = Vector{NTuple{5,Float64}}()
    mkpath(output_directory)
    history_io = open(joinpath(output_directory, "history.tsv"), "w")
    println(history_io, "iteration\telapsed_seconds\troot_lower\troot_upper\troot_gap")
    flush(history_io)
    start_time = time()
    iteration = 0

    function record_history!()
        lower = tree.V_lower[1]
        upper = tree.V_upper[1]
        row = (
            Float64(iteration),
            time() - start_time,
            lower,
            upper,
            upper - lower,
        )
        push!(history, row)
        println(history_io, join(row, '\t'))
        flush(history_io)
    end

    try
        record_history!()
        while (
            iteration < solver.max_steps
            && time() - start_time < solver.max_time
            && NativeSARSOP.root_diff(tree) > solver.precision
        )
            NativeSARSOP.sample!(solver, tree)
            NativeSARSOP.backup!(tree)
            NativeSARSOP.prune!(solver, tree)
            iteration += 1
            record_history!()
        end
    finally
        close(history_io)
    end

    alpha_vectors = hcat([collect(alpha.alpha) for alpha in tree.Γ]...)
    # AlphaVec.action indexes ordered_actions(model), whose labels are 1-based.
    # The Python-facing action map is deliberately converted back to zero-based.
    action_map = Int64[alpha.action - 1 for alpha in tree.Γ]
    write_float64(joinpath(output_directory, "alpha_vectors.bin"), alpha_vectors)
    write_int64(joinpath(output_directory, "action_map.bin"), action_map)
    write_float64(joinpath(output_directory, "corner_upper.bin"), tree.Vs_upper)

    belief_count = length(tree.b)
    if export_beliefs
        belief_matrix = hcat([collect(belief) for belief in tree.b]...)
        write_float64(joinpath(output_directory, "belief_points.bin"), belief_matrix)
        open(joinpath(output_directory, "belief_metadata.tsv"), "w") do io
            println(io, "index\tlower\tupper\tpruned\treal\tterminal")
            for index in eachindex(tree.b)
                println(
                    io,
                    join((
                        index - 1,
                        tree.V_lower[index],
                        tree.V_upper[index],
                        tree.b_pruned[index],
                        tree.is_real[index],
                        tree.is_terminal[index],
                    ), '\t'),
                )
            end
        end
    end

    root_lower = tree.V_lower[1]
    root_upper = tree.V_upper[1]
    dirac_lower = zeros(n_physical_states)
    dirac_upper = zeros(n_physical_states)
    for state in 1:n_physical_states
        belief = zeros(n_states)
        belief[state] = 1.0
        dirac_lower[state] = NativeSARSOP.lower_value(tree, belief)
        dirac_upper[state] = NativeSARSOP.upper_value(tree, belief)
    end
    write_float64(joinpath(output_directory, "dirac_lower.bin"), dirac_lower)
    write_float64(joinpath(output_directory, "dirac_upper.bin"), dirac_upper)

    stop_reason = if NativeSARSOP.root_diff(tree) <= solver.precision
        "precision"
    elseif iteration >= solver.max_steps
        "max_steps"
    else
        "max_time"
    end
    open(joinpath(output_directory, "result.tsv"), "w") do io
        println(io, "iterations\t", iteration)
        println(io, "elapsed_seconds\t", time() - start_time)
        println(io, "initialization_seconds\t", initialization_seconds)
        println(io, "initial_lower_residual\t", initial_lower_residual)
        println(io, "initial_upper_residual\t", initial_upper_residual)
        println(io, "initial_upper_iterations\t", initial_upper_iterations)
        println(io, "initial_upper_bound\t", initial_upper_method)
        println(
            io,
            "initial_lower_subsolution_shift\t",
            initial_lower_subsolution_shift,
        )
        println(io, "root_lower\t", root_lower)
        println(io, "root_upper\t", root_upper)
        println(io, "root_gap\t", root_upper - root_lower)
        println(io, "alpha_count\t", size(alpha_vectors, 2))
        println(io, "belief_count\t", belief_count)
        println(io, "stop_reason\t", stop_reason)
        println(io, "julia_version\t", VERSION)
        println(io, "native_sarsop_version\t", pkgversion(NativeSARSOP))
        println(io, "pomdps_version\t", pkgversion(POMDPs))
        println(io, "pomdp_tools_version\t", pkgversion(POMDPTools))
    end
end

length(ARGS) == 2 || error("usage: solve_jpo.jl INPUT_DIRECTORY OUTPUT_DIRECTORY")
solve_and_export(abspath(ARGS[1]), abspath(ARGS[2]))
