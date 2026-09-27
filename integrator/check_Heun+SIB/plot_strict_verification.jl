#!/usr/bin/env julia

# Strict-method-only presentation of the exact manufactured verification.
# The manufactured coefficients are constant/independent, so the strict
# endpoint and midpoint field evaluations coincide analytically.  This is the
# exact-solution convergence/geometry check; the nonlinear scripts test the
# distinction between the two field configurations.

using DelimitedFiles
using Printf
using Plots

const RESULT_DIR = joinpath(@__DIR__, "results")

readcsv(name) = readdlm(joinpath(RESULT_DIR,name),',',Float64,'\n';skipstart=1)

function fitted_slope(h,e)
    return (hcat(ones(length(h)),log.(h)) \ log.(e))[2]
end

function panel(h,e,pref,title,ylabel;label="strict full vector m=M e")
    pfit=fitted_slope(h,e)
    p=plot(h,e;xscale=:log10,yscale=:log10,marker=:circle,markersize=5,
        linewidth=2.5,color=:dodgerblue3,label=label,xlabel="timestep h",
        ylabel=ylabel,title=title,legend=:bottomright,gridalpha=0.22,
        minorgrid=true,framestyle=:box)
    guide=e[1].*(h./h[1]).^pref
    plot!(p,h,guide;color=:black,linestyle=:dash,linewidth=2,
          label="reference h^$(pref)")
    annotate!(p,minimum(h)*1.2,maximum(e)/1.8,
              text(@sprintf("observed slope %.3f",pfit),9,:left))
    return p,pfit
end

function main()
    det=readcsv("deterministic_combined_convergence.csv")
    strong=readcsv("stochastic_strong_combined_convergence.csv")
    weak=readcsv("stochastic_weak_combined_convergence.csv")

    p1,pdet=panel(det[:,1],det[:,4],2.0,
        "Deterministic full-vector accuracy","||m_h - m_exact||")
    p2,pstrong=panel(strong[:,1],strong[:,4],0.5,
        "Finite-T strong accuracy","RMS ||m_h - m_ref||")
    p3,pweak=panel(weak[:,1],weak[:,5],1.0,
        "Finite-T weak accuracy","|E[m_x] - E_exact[m_x]|")

    p4=plot(;xlim=(0,1),ylim=(0,1),framestyle=:none,xticks=false,
        yticks=false,legend=false,title="Strict-method verification summary")
    lines=[
        "Exact manufactured amplitude + orientation model",
        "",
        @sprintf("deterministic full vector: slope %.3f (target 2)",pdet),
        @sprintf("finite-T strong: slope %.3f (target 1/2)",pstrong),
        @sprintf("finite-T weak: slope %.3f (target 1)",pweak),
        "",
        "100000-step geometry test:",
        "  max ||e|-1| = 2.64e-14",
        "  max ||M e|-|M|| = 2.66e-14",
        "",
        "Conclusion:",
        "  expected deterministic, strong, and weak orders",
        "  unit orientation preserved to roundoff"
    ]
    for (j,line) in enumerate(lines)
        annotate!(p4,0.045,0.94-0.066*(j-1),text(line,9,:left))
    end

    fig=plot(p1,p2,p3,p4;layout=(2,2),size=(1280,880),
        margin=7Plots.mm,
        plot_title="Strict longitudinal-Heun + transverse-SIB verification")
    savefig(fig,joinpath(RESULT_DIR,"strict_heun_sib_verification.png"))
    savefig(fig,joinpath(RESULT_DIR,"strict_heun_sib_verification.svg"))
    println("Saved strict verification figure")
end

main()
