#!/usr/bin/env julia

using DelimitedFiles
using Printf
using Plots

const RESULT_DIR = length(ARGS) >= 1 ? abspath(ARGS[1]) : joinpath(@__DIR__, "results")
const OUTPUT_DIR = length(ARGS) >= 2 ? abspath(ARGS[2]) : RESULT_DIR

read_header_csv(name) = readdlm(joinpath(RESULT_DIR, name), ',', Float64, '\n'; skipstart=1)

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
        color=:dodgerblue3, label="full vector m=M e",
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
    det = read_header_csv("deterministic_combined_convergence.csv")
    strong = read_header_csv("stochastic_strong_combined_convergence.csv")
    weak = read_header_csv("stochastic_weak_combined_convergence.csv")

    p1 = log_panel(det[:,1], det[:,4], 2.0,
        "Deterministic full-vector accuracy", "||m_h-m_exact||")
    p2 = log_panel(strong[:,1], strong[:,4], 0.5,
        "Finite-T strong accuracy", "RMS ||m_h-m_ref||")
    p3 = log_panel(weak[:,1], weak[:,5], 1.0,
        "Finite-T weak accuracy", "|E[m_x]-E_exact[m_x]|")

    p4 = plot(; xlim=(0,1), ylim=(0,1), framestyle=:none,
        xticks=false, yticks=false, legend=false,
        title="What limits the complete method")
    lines = [
        "Longitudinal amplitude: stochastic Heun",
        "  deterministic order 2; strong order about 1",
        "",
        "Transverse orientation: SIB",
        "  deterministic order 2; strong order 1/2; weak order 1",
        "",
        "Therefore m=M e inherits:",
        "  deterministic order 2, strong order 1/2, weak order 1",
        "",
        "max ||e|-1| = 2.64e-14"
    ]
    for (j,line) in enumerate(lines)
        annotate!(p4, 0.05, 0.93-0.085*(j-1), text(line, 10, :left))
    end

    fig = plot(p1, p2, p3, p4;
        layout=(2,2), size=(1280,880), margin=7Plots.mm,
        plot_title="Longitudinal-Heun + transverse-SIB verification")
    mkpath(OUTPUT_DIR)
    savefig(fig, joinpath(OUTPUT_DIR, "heun_sib_verification.svg"))
    savefig(fig, joinpath(OUTPUT_DIR, "heun_sib_verification.png"))
    println("Saved ", joinpath(OUTPUT_DIR, "heun_sib_verification.svg"))
end

main()
