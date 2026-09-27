#!/usr/bin/env julia

# Compare the strict two-field Heun-SIB step with the cheaper shared-endpoint
# approximation for a nonlinear state-dependent field.  A very fine RK4
# solution of the continuous deterministic equations is the independent
# reference; neither discrete scheme is used to construct that reference.

using LinearAlgebra
using Printf
using DelimitedFiles

const PLOTS_AVAILABLE = try
    import Plots
    true
catch
    false
end

include("coupled_heun_sib_two_fields.jl")

const State4 = NTuple{4,Float64} # (M, ex, ey, ez)

@inline function rhs(y::State4)::State4
    M = y[1]
    e = (y[2], y[3], y[4])
    b = toy_electronic_field(scale(M, e))
    gamma_parallel = 0.7
    gamma_perp = 0.4
    dM = longitudinal_drift(e, b, gamma_parallel)
    de = add(cross3(e, b),
             scale(-gamma_perp/M, cross3(e, cross3(e, b))))
    return (dM, de[1], de[2], de[3])
end

@inline add4(a::State4, b::State4)::State4 =
    (a[1]+b[1], a[2]+b[2], a[3]+b[3], a[4]+b[4])
@inline scale4(c::Float64, a::State4)::State4 =
    (c*a[1], c*a[2], c*a[3], c*a[4])

@inline function rk4_step(y::State4, h::Float64)::State4
    k1 = rhs(y)
    k2 = rhs(add4(y, scale4(0.5*h, k1)))
    k3 = rhs(add4(y, scale4(0.5*h, k2)))
    k4 = rhs(add4(y, scale4(h, k3)))
    return add4(y, scale4(h/6.0,
                          add4(add4(k1, scale4(2.0, k2)),
                               add4(scale4(2.0, k3), k4))))
end

function rk4_reference(y0::State4, T::Float64; nsteps::Int=2^19)::State4
    h = T/nsteps
    y = y0
    for _ in 1:nsteps
        y = rk4_step(y, h)
    end
    return y
end

@inline physical_vector(y::State4)::Vec3 = scale(y[1], (y[2],y[3],y[4]))

function discrete_solution(stepper, M0, e0, T, nsteps)
    h = T/nsteps
    M, e = M0, e0
    zero_noise = (0.0, 0.0, 0.0)
    for _ in 1:nsteps
        M, e = stepper(M, e, h, 0.0, zero_noise, toy_electronic_field;
                       kT=0.0)
    end
    return M, e
end

function fit_order(h, err)
    A = hcat(ones(length(h)), log.(h))
    return (A \ log.(err))[2]
end

function main()
    T = 1.0
    M0 = 0.45
    e0raw = (1.0, 0.35, -0.20)
    e0 = scale(1.0/norm3(e0raw), e0raw)
    y0 = (M0, e0[1], e0[2], e0[3])

    @printf("Building independent RK4 reference ...\n")
    yref = rk4_reference(y0, T)
    mref = physical_vector(yref)

    Ns = 2 .^ collect(3:11)
    hs = T ./ Ns
    strict_error = Float64[]
    shared_error = Float64[]

    for N in Ns
        Ms, es = discrete_solution(strict_step, M0, e0, T, N)
        Ma, ea = discrete_solution(shared_endpoint_step, M0, e0, T, N)
        push!(strict_error, norm3(sub(scale(Ms, es), mref)))
        push!(shared_error, norm3(sub(scale(Ma, ea), mref)))
    end

    # Fit the asymptotic five finest steps.
    fine = length(hs)-4:length(hs)
    p_strict = fit_order(hs[fine], strict_error[fine])
    p_shared = fit_order(hs[fine], shared_error[fine])

    outdir = joinpath(@__DIR__, "results")
    mkpath(outdir)
    open(joinpath(outdir, "strict_vs_shared.csv"), "w") do io
        println(io, "h,strict_error,shared_endpoint_error")
        writedlm(io, hcat(hs, strict_error, shared_error), ',')
    end
    open(joinpath(outdir, "strict_vs_shared_summary.txt"), "w") do io
        @printf(io, "strict fitted order = %.6f\n", p_strict)
        @printf(io, "shared fitted order = %.6f\n", p_shared)
        @printf(io, "finest strict error = %.12e\n", strict_error[end])
        @printf(io, "finest shared error = %.12e\n", shared_error[end])
        @printf(io, "finest error ratio shared/strict = %.6f\n",
                shared_error[end]/strict_error[end])
    end

    @printf("strict fitted order              = %.6f\n", p_strict)
    @printf("shared-endpoint fitted order     = %.6f\n", p_shared)
    @printf("finest error ratio shared/strict = %.3f\n",
            shared_error[end]/strict_error[end])

    if PLOTS_AVAILABLE
        p = Plots.plot(hs, strict_error;
            xscale=:log10, yscale=:log10, marker=:circle, linewidth=2,
            label=@sprintf("strict: slope %.3f", p_strict),
            xlabel="timestep h", ylabel="final full-vector error",
            title="State-dependent field: strict vs shared endpoint",
            legend=:topleft, framestyle=:box)
        Plots.plot!(p, hs, shared_error; marker=:diamond, linewidth=2,
                    label=@sprintf("shared endpoint: slope %.3f", p_shared))
        Plots.savefig(p, joinpath(outdir, "strict_vs_shared.png"))
        Plots.savefig(p, joinpath(outdir, "strict_vs_shared.svg"))
    else
        @warn "Plots.jl unavailable; CSV and summary were still written"
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
