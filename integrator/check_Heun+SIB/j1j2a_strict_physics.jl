#!/usr/bin/env julia

# Physics-facing benchmark of the strict three-field Heun+SIB integrator for
# the square-lattice soft-spin J1-J2-a model.  It checks the Neel/stripe ground-
# state crossing and the analytically predicted relaxed spin amplitudes.

using Random, LinearAlgebra, Statistics, Printf, DelimitedFiles
const PLOTS_AVAILABLE = try; import Plots; true; catch; false; end

Base.@kwdef struct P
    L::Int=12; J1::Float64=1.0; J2::Float64=0.3
    a::Float64=40.0; b::Float64=0.05
    dt::Float64=0.005; steps::Int=4000
    gamma_parallel::Float64=1.0; gamma_perp::Float64=1.0
end
@inline lm(i,L)=i==1 ? L : i-1
@inline lp(i,L)=i==L ? 1 : i+1

mutable struct W
    M::Matrix{Float64}; Mp::Matrix{Float64}; e::Array{Float64,3}
    ep::Array{Float64,3}; em::Array{Float64,3}; en::Array{Float64,3}
    mp::Array{Float64,3}; mm::Array{Float64,3}
    b0::Array{Float64,3}; bh::Array{Float64,3}; bs::Array{Float64,3}
end
W(L)=W(zeros(L,L),zeros(L,L),(zeros(3,L,L) for _ in 1:9)...)

function field!(out,m,p)
    L=p.L
    for y in 1:L,x in 1:L
        xm,xp=lm(x,L),lp(x,L); ym,yp=lm(y,L),lp(y,L)
        r2=sum(abs2,@view m[:,x,y]); radial=p.a*(r2-p.b)
        for c in 1:3
            nn=m[c,xm,y]+m[c,xp,y]+m[c,x,ym]+m[c,x,yp]
            nnn=m[c,xm,ym]+m[c,xm,yp]+m[c,xp,ym]+m[c,xp,yp]
            out[c,x,y]=-p.J1*nn-p.J2*nnn-radial*m[c,x,y]
        end
    end
end

function energy(m,p)
    E=0.0
    for y in 1:p.L,x in 1:p.L
        xp,yp=lp(x,p.L),lp(y,p.L)
        E+=p.J1*dot(@view(m[:,x,y]),@view(m[:,xp,y])+@view(m[:,x,yp]))
        E+=p.J2*dot(@view(m[:,x,y]),@view(m[:,xp,yp])+@view(m[:,xp,lm(y,p.L)]))
        r2=sum(abs2,@view m[:,x,y]); E+=0.25p.a*(r2-p.b)^2
    end
    E
end

@inline function cayley!(out,base,qx,qy,qz,x,y)
    wx,wy,wz=0.5qx,0.5qy,0.5qz; w2=wx^2+wy^2+wz^2
    ex,ey,ez=base[1,x,y],base[2,x,y],base[3,x,y]
    d=wx*ex+wy*ey+wz*ez
    cx=ey*wz-ez*wy; cy=ez*wx-ex*wz; cz=ex*wy-ey*wx; den=1+w2
    out[1,x,y]=((1-w2)*ex+2cx+2wx*d)/den
    out[2,x,y]=((1-w2)*ey+2cy+2wy*d)/den
    out[3,x,y]=((1-w2)*ez+2cz+2wz*d)/den
end

function decompose!(w,m,p)
    for y in 1:p.L,x in 1:p.L
        r=norm(@view m[:,x,y]); w.M[x,y]=r; @views w.e[:,x,y].=m[:,x,y]./r
    end
end

function strict_step!(m,w,p)
    decompose!(w,m,p); field!(w.b0,m,p)                       # field 1: old
    for y in 1:p.L,x in 1:p.L
        f0=dot(@view(w.e[:,x,y]),@view(w.b0[:,x,y]))
        w.Mp[x,y]=w.M[x,y]+p.gamma_parallel*p.dt*f0
        ex,ey,ez=w.e[1,x,y],w.e[2,x,y],w.e[3,x,y]
        bx,by,bz=w.b0[1,x,y],w.b0[2,x,y],w.b0[3,x,y]
        cx=ey*bz-ez*by; cy=ez*bx-ex*bz; cz=ex*by-ey*bx
        cayley!(w.ep,w.e,p.dt*(bx-p.gamma_perp*cx/w.M[x,y]),
                p.dt*(by-p.gamma_perp*cy/w.M[x,y]),
                p.dt*(bz-p.gamma_perp*cz/w.M[x,y]),x,y)
        @views w.mp[:,x,y].=w.Mp[x,y].*w.ep[:,x,y]
    end
    field!(w.bh,w.mp,p)                                      # field 2: Heun endpoint
    for y in 1:p.L,x in 1:p.L
        Mm=0.5*(w.M[x,y]+w.Mp[x,y])
        @views w.em[:,x,y].=0.5.*(w.e[:,x,y].+w.ep[:,x,y])
        @views w.mm[:,x,y].=Mm.*w.em[:,x,y]
    end
    field!(w.bs,w.mm,p)                                      # field 3: SIB midpoint
    for y in 1:p.L,x in 1:p.L
        f0=dot(@view(w.e[:,x,y]),@view(w.b0[:,x,y]))
        fp=dot(@view(w.ep[:,x,y]),@view(w.bh[:,x,y]))
        Mn=w.M[x,y]+0.5p.gamma_parallel*p.dt*(f0+fp)
        Mm=0.5*(w.M[x,y]+w.Mp[x,y]); ex,ey,ez=w.em[1,x,y],w.em[2,x,y],w.em[3,x,y]
        bx,by,bz=w.bs[1,x,y],w.bs[2,x,y],w.bs[3,x,y]
        cx=ey*bz-ez*by; cy=ez*bx-ex*bz; cz=ex*by-ey*bx
        cayley!(w.en,w.e,p.dt*(bx-p.gamma_perp*cx/Mm),
                p.dt*(by-p.gamma_perp*cy/Mm),
                p.dt*(bz-p.gamma_perp*cz/Mm),x,y)
        @views m[:,x,y].=Mn.*w.en[:,x,y]
    end
end

function initialize(p,phase,seed)
    rng=MersenneTwister(seed); m=zeros(3,p.L,p.L)
    for y in 1:p.L,x in 1:p.L
        sign=phase==:neel ? (-1)^(x+y) : (-1)^x
        v=[0.18randn(rng),0.18randn(rng),sign+0.18randn(rng)]; v./=norm(v)
        m[:,x,y].=0.20.*v
    end
    m
end

function order(m,p,qx,qy)
    v=zeros(3)
    for y in 1:p.L,x in 1:p.L
        s=(-1)^(qx*(x-1)+qy*(y-1)); @views v.+=s.*m[:,x,y]
    end
    norm(v)/p.L^2
end

function analytic(p,phase)
    M2=phase==:neel ? p.b+4*(p.J1-p.J2)/p.a : p.b+4p.J2/p.a
    M=sqrt(max(M2,0)); c=phase==:neel ? (-2p.J1+2p.J2) : -2p.J2
    E=c*M2+.25p.a*(M2-p.b)^2
    M,E
end

function run_branch(p,phase)
    m=initialize(p,phase,314+Int(round(100p.J2))+(phase==:stripe ? 1000 : 0)); w=W(p.L)
    Ehist=Float64[energy(m,p)/p.L^2]
    for k in 1:p.steps
        strict_step!(m,w,p)
        k%100==0 && push!(Ehist,energy(m,p)/p.L^2)
    end
    lens=[norm(@view m[:,x,y]) for x in 1:p.L,y in 1:p.L]
    return mean(lens),energy(m,p)/p.L^2,order(m,p,1,1),max(order(m,p,1,0),order(m,p,0,1)),Ehist
end

function main(args=ARGS)
    quick="--quick" in args; ratios=collect(0.1:0.1:0.9)
    rows=Vector{NTuple{9,Float64}}(); histories=Dict()
    for r in ratios, (pid,ph) in enumerate((:neel,:stripe))
        p=P(J2=r,steps=quick ? 800 : 4000,L=quick ? 8 : 12)
        M,E,Qn,Qs,hist=run_branch(p,ph); Ma,Ea=analytic(p,ph)
        push!(rows,(r,pid,M,Ma,E,Ea,Qn,Qs,maximum(diff(hist))))
        (r==0.3 || r==0.7) && (histories[(r,ph)]=hist)
        @printf("J2/J1=%.1f %-6s M %.4f/%.4f E %.5f/%.5f Qn %.3f Qs %.3f\n",r,String(ph),M,Ma,E,Ea,Qn,Qs)
    end
    data=reduce(vcat,permutedims.(collect.(rows))); out=joinpath(@__DIR__,"results"); mkpath(out)
    open(joinpath(out,"j1j2a_strict_physics.csv"),"w") do io
        println(io,"J2_over_J1,phase_id,M_numeric,M_exact,E_numeric,E_exact,Q_neel,Q_stripe,max_energy_increase")
        writedlm(io,data,',')
    end
    PLOTS_AVAILABLE || return
    n=data[data[:,2].==1,:]; s=data[data[:,2].==2,:]
    p1=Plots.plot(n[:,1],n[:,5],marker=:circle,label="started near Neel",xlabel="J2/J1",ylabel="relaxed energy/site",title="Ground-state branches",framestyle=:box)
    Plots.plot!(p1,n[:,1],n[:,6],linestyle=:dash,label="Neel analytic"); Plots.plot!(p1,s[:,1],s[:,5],marker=:square,label="started near stripe"); Plots.plot!(p1,s[:,1],s[:,6],linestyle=:dash,label="stripe analytic"); Plots.vline!(p1,[0.5],color=:black,linestyle=:dot,label="J2/J1=1/2")
    p2=Plots.plot(n[:,1],n[:,3],marker=:circle,label="started near Neel",xlabel="J2/J1",ylabel="relaxed mean |m|",title="Soft-spin amplitude",framestyle=:box)
    Plots.plot!(p2,n[:,1],n[:,4],linestyle=:dash,label="Neel analytic"); Plots.plot!(p2,s[:,1],s[:,3],marker=:square,label="started near stripe"); Plots.plot!(p2,s[:,1],s[:,4],linestyle=:dash,label="stripe analytic")
    p3=Plots.plot(n[:,1],n[:,7],marker=:circle,label="Q(pi,pi), Neel-start run",xlabel="J2/J1",ylabel="order parameter",title="Magnetic character",framestyle=:box)
    Plots.plot!(p3,s[:,1],s[:,8],marker=:square,label="max stripe Q, stripe-start run")
    fig=Plots.plot(p1,p2,p3,layout=(1,3),size=(1500,430),margin=5Plots.mm,plot_title="Strict three-field Heun+SIB: soft-spin J1-J2-a physics benchmark")
    Plots.savefig(fig,joinpath(out,"j1j2a_strict_physics.png")); Plots.savefig(fig,joinpath(out,"j1j2a_strict_physics.svg"))
end
main()
