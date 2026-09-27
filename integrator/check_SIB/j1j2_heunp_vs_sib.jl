#!/usr/bin/env julia

# Physical fixed-length benchmark for HeunP and SIB on the square-lattice
# classical J1-J2 model.  The deterministic precession dynamics conserve the
# Hamiltonian and every spin length in continuous time.  We measure the signed
# time-averaged energy drift per site and the maximum spin-length error versus h.

using Random
using Statistics
using LinearAlgebra
using Printf
using DelimitedFiles

const PLOTS_AVAILABLE = try
    import Plots
    true
catch
    false
end

Base.@kwdef struct Params
    L::Int = 16
    J1::Float64 = 1.0
    J2::Float64 = 0.30
    total_time::Float64 = 20.0
    sample_dt::Float64 = 0.20
    seeds::Int = 8
end

@inline left(i,L)=i==1 ? L : i-1
@inline right(i,L)=i==L ? 1 : i+1

function initialize_spins(L,seed)
    rng=MersenneTwister(seed)
    s=zeros(Float64,3,L,L)
    for y in 1:L,x in 1:L
        v=randn(rng,3); v./=norm(v)
        s[:,x,y].=v
    end
    return s
end

function field!(b,s,p)
    L=p.L
    for y in 1:L,x in 1:L
        xm,xp=left(x,L),right(x,L); ym,yp=left(y,L),right(y,L)
        for c in 1:3
            nn=s[c,xm,y]+s[c,xp,y]+s[c,x,ym]+s[c,x,yp]
            nnn=s[c,xm,ym]+s[c,xm,yp]+s[c,xp,ym]+s[c,xp,yp]
            b[c,x,y]=-p.J1*nn-p.J2*nnn
        end
    end
end

function energy(s,p)
    E=0.0; L=p.L
    for y in 1:L,x in 1:L
        xp,yp=right(x,L),right(y,L)
        for c in 1:3
            E += p.J1*s[c,x,y]*(s[c,xp,y]+s[c,x,yp])
            E += p.J2*s[c,x,y]*(s[c,xp,yp]+s[c,xp,left(y,L)])
        end
    end
    return E
end

function torque!(f,s,b,p)
    for y in 1:p.L,x in 1:p.L
        sx,sy,sz=s[1,x,y],s[2,x,y],s[3,x,y]
        bx,by,bz=b[1,x,y],b[2,x,y],b[3,x,y]
        f[1,x,y]=sy*bz-sz*by
        f[2,x,y]=sz*bx-sx*bz
        f[3,x,y]=sx*by-sy*bx
    end
end

@inline function cayley_site!(out,base,q,x,y)
    wx,wy,wz=0.5*q[1,x,y],0.5*q[2,x,y],0.5*q[3,x,y]
    ex,ey,ez=base[1,x,y],base[2,x,y],base[3,x,y]
    w2=wx^2+wy^2+wz^2; d=wx*ex+wy*ey+wz*ez
    cx=ey*wz-ez*wy; cy=ez*wx-ex*wz; cz=ex*wy-ey*wx
    den=1+w2
    out[1,x,y]=((1-w2)*ex+2cx+2wx*d)/den
    out[2,x,y]=((1-w2)*ey+2cy+2wy*d)/den
    out[3,x,y]=((1-w2)*ez+2cz+2wz*d)/den
end

mutable struct Work
    b0::Array{Float64,3}; bp::Array{Float64,3}
    f0::Array{Float64,3}; fp::Array{Float64,3}
    pred::Array{Float64,3}; mid::Array{Float64,3}; q::Array{Float64,3}
end
Work(L)=Work((zeros(3,L,L) for _ in 1:7)...)

function heunp_step!(s,w,p,h)
    field!(w.b0,s,p); torque!(w.f0,s,w.b0,p)
    @. w.pred=s+h*w.f0
    field!(w.bp,w.pred,p); torque!(w.fp,w.pred,w.bp,p)
    @. w.pred=s+0.5h*(w.f0+w.fp)
    for y in 1:p.L,x in 1:p.L
        r=sqrt(sum(abs2,@view w.pred[:,x,y]))
        @views s[:,x,y].=w.pred[:,x,y]./r
    end
end

function sib_step!(s,w,p,h)
    # Eq. (18) predictor: coefficients at the old configuration.
    field!(w.b0,s,p); @. w.q=h*w.b0
    for y in 1:p.L,x in 1:p.L
        cayley_site!(w.pred,s,w.q,x,y)
    end
    # Eq. (18) corrector: coefficients at the predicted midpoint.
    @. w.mid=0.5*(s+w.pred)
    field!(w.bp,w.mid,p); @. w.q=h*w.bp
    for y in 1:p.L,x in 1:p.L
        cayley_site!(w.pred,s,w.q,x,y)
    end
    s.=w.pred
end

function max_norm_error(s)
    err=0.0
    for y in axes(s,3),x in axes(s,2)
        err=max(err,abs(sqrt(sum(abs2,@view s[:,x,y]))-1.0))
    end
    return err
end

function run_one(method,h,p,seed)
    s=initialize_spins(p.L,seed); w=Work(p.L); E0=energy(s,p)
    nsteps=round(Int,p.total_time/h); stride=max(1,round(Int,p.sample_dt/h))
    drift=Float64[]; normerr=0.0
    for step in 1:nsteps
        method(s,w,p,h)
        normerr=max(normerr,max_norm_error(s))
        if step%stride==0 || step==nsteps
            # Signed error convention chosen to match the Mentink-style plot:
            # positive means the numerical mean energy lies below E(0).
            push!(drift,(E0-energy(s,p))/(p.L^2*p.J1))
        end
    end
    return mean(drift),normerr
end

function main(args=ARGS)
    quick="--quick" in args
    p=quick ? Params(L=10,total_time=5.0,seeds=3) : Params()
    hs=quick ? [0.005,0.01,0.02,0.04,0.08,0.12] :
               [0.0025,0.005,0.01,0.02,0.04,0.06,0.08,0.10,0.125,0.15,0.20,0.25]
    methods=[("HeunP",heunp_step!), ("SIB",sib_step!)]
    rows=Vector{NTuple{5,Float64}}()
    for (mi,(name,method)) in enumerate(methods)
        for h in hs
            ed=Float64[]; ne=Float64[]
            for seed in 1:p.seeds
                a,b=run_one(method,h,p,10_000+seed); push!(ed,a); push!(ne,b)
            end
            push!(rows,(mi,h,mean(ed),std(ed)/sqrt(length(ed)),maximum(ne)))
            @printf("%-5s h=%7.4f  mean dE/NJ1=% .4e  max norm err=%.3e\n",
                    name,h,mean(ed),maximum(ne))
        end
    end
    outdir=joinpath(@__DIR__,"results"); mkpath(outdir)
    data=reduce(vcat,permutedims.(collect.(rows)))
    open(joinpath(outdir,"j1j2_heunp_vs_sib.csv"),"w") do io
        println(io,"method_id,h,mean_energy_error_per_site,standard_error,max_norm_error")
        writedlm(io,data,',')
    end
    PLOTS_AVAILABLE || return
    he=data[data[:,1].==1,:]; si=data[data[:,1].==2,:]
    p1=Plots.plot(he[:,2],he[:,3],yerror=he[:,4],marker=:circle,
        linestyle=:dash,color=:green,label="HeunP",xlabel="step size h (units of J1^-1)",
        ylabel="error in mean energy / (N J1)",title="Energy stability",framestyle=:box)
    Plots.plot!(p1,si[:,2],si[:,3],yerror=si[:,4],marker=:square,
        linestyle=:dash,color=:blue,label="SIB")
    Plots.hline!(p1,[0.0],color=:black,linewidth=1,label=false)
    p2=Plots.plot(he[:,2],he[:,5],marker=:circle,linestyle=:dash,
        color=:green,label="HeunP (post-projection)",xlabel="step size h",
        ylabel="max_i,t |||S_i||-1|",yscale=:log10,title="Spin-length preservation",
        framestyle=:box)
    Plots.plot!(p2,si[:,2],si[:,5],marker=:square,linestyle=:dash,
        color=:blue,label="SIB (intrinsic)")
    fig=Plots.plot(p1,p2,layout=(1,2),size=(1150,450),margin=5Plots.mm,
        plot_title=@sprintf("Square-lattice J1-J2 model: J2/J1=%.2f, L=%d",p.J2/p.J1,p.L))
    Plots.savefig(fig,joinpath(outdir,"j1j2_heunp_vs_sib.png"))
    Plots.savefig(fig,joinpath(outdir,"j1j2_heunp_vs_sib.svg"))
end

main()
