#!/usr/bin/env julia

# Three-part comparison of the strict two-field and shared-endpoint schemes:
# deterministic full-vector error, finite-T strong error, and finite-T weak
# error.  Coarse and reference trajectories share exactly the same Brownian
# paths.  Run with --quick for a shorter diagnostic calculation.

using Random
using Printf
using DelimitedFiles

include("compare_strict_vs_shared.jl")

@inline observable(M::Float64, e::Vec3) = M*e[1] + 0.2*M^2

function integrate_increments(stepper, M0, e0, h, dWp, dWx, dWy, dWz;
                              block=1, kT=1.0e-3)
    M, e = M0, e0
    n = length(dWp) ÷ block
    for j in 1:n
        first = (j-1)*block + 1
        last = j*block
        swp = 0.0; swx = 0.0; swy = 0.0; swz = 0.0
        @inbounds for k in first:last
            swp += dWp[k]; swx += dWx[k]; swy += dWy[k]; swz += dWz[k]
        end
        M, e = stepper(M, e, h, swp, (swx,swy,swz),
                       toy_electronic_field; kT=kT)
    end
    return M, e
end

function stochastic_comparison(; npaths=2000, nref=16384, seed=71129,
                               kT=1.0e-3)
    rng = MersenneTwister(seed)
    T = 1.0
    M0 = 0.45
    e0raw = (1.0, 0.35, -0.20)
    e0 = scale(1.0/norm3(e0raw), e0raw)
    maxpow = min(10, round(Int,log2(nref))-3)
    Ns = 2 .^ collect(5:maxpow)
    hs = T ./ Ns
    dtref = T/nref

    sumsq_strict = zeros(length(Ns))
    sumsq_shared = zeros(length(Ns))
    sumweak_strict = zeros(length(Ns))
    sumweak_shared = zeros(length(Ns))
    sumweak2_strict = zeros(length(Ns))
    sumweak2_shared = zeros(length(Ns))

    dWp = Vector{Float64}(undef,nref)
    dWx = similar(dWp); dWy = similar(dWp); dWz = similar(dWp)
    rootdt = sqrt(dtref)

    for path in 1:npaths
        randn!(rng,dWp); randn!(rng,dWx); randn!(rng,dWy); randn!(rng,dWz)
        dWp .*= rootdt; dWx .*= rootdt; dWy .*= rootdt; dWz .*= rootdt

        Mr, er = integrate_increments(strict_step,M0,e0,dtref,
                                      dWp,dWx,dWy,dWz;kT=kT)
        mr = scale(Mr,er)
        phir = observable(Mr,er)

        for (j,N) in enumerate(Ns)
            block = nref ÷ N
            Ms,es = integrate_increments(strict_step,M0,e0,T/N,
                                         dWp,dWx,dWy,dWz;block=block,kT=kT)
            Ma,ea = integrate_increments(shared_endpoint_step,M0,e0,T/N,
                                         dWp,dWx,dWy,dWz;block=block,kT=kT)
            ds = norm3(sub(scale(Ms,es),mr))
            da = norm3(sub(scale(Ma,ea),mr))
            sumsq_strict[j] += ds^2
            sumsq_shared[j] += da^2
            ws = observable(Ms,es)-phir
            wa = observable(Ma,ea)-phir
            sumweak_strict[j] += ws
            sumweak_shared[j] += wa
            sumweak2_strict[j] += ws^2
            sumweak2_shared[j] += wa^2
        end
    end

    strong_strict = sqrt.(sumsq_strict ./ npaths)
    strong_shared = sqrt.(sumsq_shared ./ npaths)
    meanws = sumweak_strict ./ npaths
    meanwa = sumweak_shared ./ npaths
    weak_strict = abs.(meanws)
    weak_shared = abs.(meanwa)
    se_strict = sqrt.(max.(sumweak2_strict./npaths .- meanws.^2,0.0)./npaths)
    se_shared = sqrt.(max.(sumweak2_shared./npaths .- meanwa.^2,0.0)./npaths)
    return hs,strong_strict,strong_shared,weak_strict,weak_shared,se_strict,se_shared
end

function deterministic_data()
    T=1.0; M0=0.45
    e0raw=(1.0,0.35,-0.20); e0=scale(1.0/norm3(e0raw),e0raw)
    y0=(M0,e0[1],e0[2],e0[3]); mref=physical_vector(rk4_reference(y0,T))
    Ns=2 .^ collect(3:11); hs=T./Ns
    es=Float64[]; ea=Float64[]
    for N in Ns
        Ms,vs=discrete_solution(strict_step,M0,e0,T,N)
        Ma,va=discrete_solution(shared_endpoint_step,M0,e0,T,N)
        push!(es,norm3(sub(scale(Ms,vs),mref)))
        push!(ea,norm3(sub(scale(Ma,va),mref)))
    end
    return hs,es,ea
end

function run_all(; quick=false)
    npaths = quick ? 400 : 2000
    nref = quick ? 2048 : 16384
    hd,ed_s,ed_a = deterministic_data()
    hs,estr_s,estr_a,ew_s,ew_a,se_s,se_a =
        stochastic_comparison(npaths=npaths,nref=nref)

    pd_s=fit_order(hd[end-4:end],ed_s[end-4:end])
    pd_a=fit_order(hd[end-4:end],ed_a[end-4:end])
    ps_s=fit_order(hs[end-3:end],estr_s[end-3:end])
    ps_a=fit_order(hs[end-3:end],estr_a[end-3:end])
    resolved_s = all(ew_s .> 2.0 .* se_s)
    resolved_a = all(ew_a .> 2.0 .* se_a)
    pw_s = resolved_s ? fit_order(hs,ew_s) : NaN
    pw_a = resolved_a ? fit_order(hs,ew_a) : NaN

    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    open(joinpath(outdir,"strict_vs_shared_stochastic.csv"),"w") do io
        println(io,"h,strong_strict,strong_shared,weak_strict,weak_shared,weak_se_strict,weak_se_shared")
        writedlm(io,hcat(hs,estr_s,estr_a,ew_s,ew_a,se_s,se_a),',')
    end
    open(joinpath(outdir,"strict_vs_shared_full_summary.txt"),"w") do io
        @printf(io,"paths = %d\nreference steps = %d\n",npaths,nref)
        @printf(io,"deterministic strict/shared = %.6f %.6f\n",pd_s,pd_a)
        @printf(io,"strong strict/shared = %.6f %.6f\n",ps_s,ps_a)
        @printf(io,"weak strict/shared = %.6f %.6f\n",pw_s,pw_a)
    end
    @printf("deterministic strict/shared: %.3f / %.3f\n",pd_s,pd_a)
    @printf("strong strict/shared:        %.3f / %.3f\n",ps_s,ps_a)
    @printf("weak strict/shared:          %s / %s\n",
            resolved_s ? @sprintf("%.3f",pw_s) : "below MC resolution",
            resolved_a ? @sprintf("%.3f",pw_a) : "below MC resolution")

    PLOTS_AVAILABLE || return
    p1=Plots.plot(hd,ed_s,xscale=:log10,yscale=:log10,marker=:circle,label=@sprintf("strict %.3f",pd_s),xlabel="h",ylabel="full-vector error",title="Deterministic",framestyle=:box)
    Plots.plot!(p1,hd,ed_a,marker=:diamond,label=@sprintf("shared %.3f",pd_a))
    p2=Plots.plot(hs,estr_s,xscale=:log10,yscale=:log10,marker=:circle,label=@sprintf("strict %.3f",ps_s),xlabel="h",ylabel="RMS path error",title=@sprintf("Finite T strong (%d paths)",npaths),framestyle=:box)
    Plots.plot!(p2,hs,estr_a,marker=:diamond,label=@sprintf("shared %.3f",ps_a))
    # Clip visual error bars on the logarithmic axis; the unmodified standard
    # errors are written to CSV.  A slope is shown only when every bias exceeds
    # two Monte Carlo standard errors.
    plotse_s=min.(se_s,0.8 .* ew_s); plotse_a=min.(se_a,0.8 .* ew_a)
    label_s=resolved_s ? @sprintf("strict %.3f",pw_s) : "strict: below MC resolution"
    label_a=resolved_a ? @sprintf("shared %.3f",pw_a) : "shared: below MC resolution"
    p3=Plots.plot(hs,ew_s,yerror=plotse_s,xscale=:log10,yscale=:log10,marker=:circle,label=label_s,xlabel="h",ylabel="weak bias",title="Finite T weak",framestyle=:box)
    Plots.plot!(p3,hs,ew_a,yerror=plotse_a,marker=:diamond,label=label_a)
    fig=Plots.plot(p1,p2,p3,layout=(1,3),size=(1500,430),margin=4Plots.mm)
    Plots.savefig(fig,joinpath(outdir,"strict_vs_shared_full.png"))
    Plots.savefig(fig,joinpath(outdir,"strict_vs_shared_full.svg"))
end

run_all(quick=("--quick" in ARGS))
