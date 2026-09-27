#!/usr/bin/env julia

using DelimitedFiles
using Printf
using Plots

const RESULT_DIR = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "results")
const OUTPUT_DIR = length(ARGS) >= 2 ? abspath(ARGS[2]) : RESULT_DIR

function read_header_csv(name)
    return readdlm(joinpath(RESULT_DIR, name), ',', Float64, '\n'; skipstart=1)
end

function slope(h, err)
    x = log.(h)
    y = log.(err)
    return (hcat(ones(length(x)), x) \ y)[2]
end

function log_panel(h, err, expected, title, ylabel)
    observed = slope(h, err)
    p = plot(h, err;
        xscale=:log10, yscale=:log10,
        marker=:circle, markersize=5, linewidth=2.4,
        color=:dodgerblue3, label="measured",
        xlabel="timestep h", ylabel=ylabel, title=title,
        legend=:bottomright, gridalpha=0.22, minorgrid=true)
    reference = err[1] .* (h ./ h[1]).^expected
    plot!(p, h, reference;
        color=:black, linestyle=:dash, linewidth=2,
        label="reference h^$(expected)")
    annotate!(p, minimum(h)*1.2, maximum(err)/1.8,
        text(@sprintf("observed slope %.3f", observed), 9, :left))
    return p
end

function main()
    det = readdlm(joinpath(RESULT_DIR, "deterministic_convergence.csv"), ',', Float64)
    strong = read_header_csv("stochastic_convergence.csv")
    weak = read_header_csv("exact_weak_convergence.csv")
    growth = read_header_csv("deterministic_error_growth.csv")

    p1 = log_panel(det[:,1], det[:,2], 2.0,
        "Deterministic endpoint accuracy", "||X_h(T)-X_exact(T)||")
    p2 = log_panel(strong[:,2], strong[:,3], 0.5,
        "Finite-T strong accuracy", "RMS path error")
    p3 = log_panel(weak[:,1], weak[:,4], 1.0,
        "Finite-T weak accuracy", "|E[X_x]-E_exact[X_x]|")

    p4 = plot(growth[:,1], growth[:,2];
        linewidth=2.4, color=:dodgerblue3, label="vector error",
        xlabel="time", ylabel="error", title="Cayley phase-error growth",
        legend=:topleft, gridalpha=0.22)
    plot!(p4, growth[:,1], growth[:,3];
        color=:black, linestyle=:dash, linewidth=2,
        label="exact phase difference")
    plot!(p4, growth[:,1], growth[:,4];
        color=:gray40, linestyle=:dot, linewidth=2,
        label="leading t h^2 / 12")

    fig = plot(p1, p2, p3, p4;
        layout=(2,2), size=(1280,880), margin=7Plots.mm,
        plot_title="Standalone SIB verification")
    mkpath(OUTPUT_DIR)
    savefig(fig, joinpath(OUTPUT_DIR, "sib_verification.svg"))
    savefig(fig, joinpath(OUTPUT_DIR, "sib_verification.png"))
    println("Saved ", joinpath(OUTPUT_DIR, "sib_verification.svg"))
end

main()
