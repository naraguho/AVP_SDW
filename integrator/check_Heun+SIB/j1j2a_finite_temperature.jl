#!/usr/bin/env julia

# Finite-temperature equilibrium demonstration for the variable-amplitude
# square-lattice J1-J2-a model, integrated with strict three-field Heun+SIB.

using Random, LinearAlgebra, Statistics, Printf, DelimitedFiles
const PLOTS_AVAILABLE = try; import Plots; true; catch; false; end

Base.@kwdef struct P
    L::Int=8; J1::Float64=1.0; J2::Float64=0.30
    a::Float64=40.0; b::Float64=0.05; temperature::Float64=0.005
    dt::Float64=0.002; steps::Int=60_000; burn::Int=20_000; sample_every::Int=20
    gamma_parallel::Float64=1.0; gamma_perp::Float64=1.0
end

@inline lm(i,L)=i==1 ? L : i-1
@inline lp(i,L)=i==L ? 1 : i+1

mutable struct W
    M::Matrix{Float64}; Mp::Matrix{Float64}; e::Array{Float64,3}
    ep::Array{Float64,3}; em::Array{Float64,3}; en::Array{Float64,3}
    mp::Array{Float64,3}; mm::Array{Float64,3}
    b0::Array{Float64,3}; bh::Array{Float64,3}; bs::Array{Float64,3}
    dWp::Matrix{Float64}; dWx::Matrix{Float64}; dWy::Matrix{Float64}; dWz::Matrix{Float64}
end
W(L)=W(zeros(L,L),zeros(L,L),(zeros(3,L,L) for _ in 1:9)...,
       zeros(L,L),zeros(L,L),zeros(L,L),zeros(L,L))

function field!(out,m,p)
    for y in 1:p.L,x in 1:p.L
        xm,xp=lm(x,p.L),lp(x,p.L); ym,yp=lm(y,p.L),lp(y,p.L)
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
        r=norm(@view m[:,x,y])
        r>1e-10 || error("spin amplitude approached zero; reduce temperature or timestep")
        w.M[x,y]=r; @views w.e[:,x,y].=m[:,x,y]./r
    end
end

function strict_step!(m,w,p,rng)
    decompose!(w,m,p); field!(w.b0,m,p)
    randn!(rng,w.dWp); randn!(rng,w.dWx); randn!(rng,w.dWy); randn!(rng,w.dWz)
    w.dWp .*= sqrt(p.dt); w.dWx .*= sqrt(p.dt); w.dWy .*= sqrt(p.dt); w.dWz .*= sqrt(p.dt)
    sigM=sqrt(2p.temperature*p.gamma_parallel)

    for y in 1:p.L,x in 1:p.L
        M=w.M[x,y]; ex,ey,ez=w.e[1,x,y],w.e[2,x,y],w.e[3,x,y]
        bx,by,bz=w.b0[1,x,y],w.b0[2,x,y],w.b0[3,x,y]
        f0=ex*bx+ey*by+ez*bz
        w.Mp[x,y]=M+p.gamma_parallel*p.dt*f0+sigM*w.dWp[x,y]
        cx=ey*bz-ez*by; cy=ez*bx-ex*bz; cz=ex*by-ey*bx
        sigE=sqrt(2p.temperature*p.gamma_perp)/M
        qx=p.dt*(bx-p.gamma_perp*cx/M)-sigE*w.dWx[x,y]
        qy=p.dt*(by-p.gamma_perp*cy/M)-sigE*w.dWy[x,y]
        qz=p.dt*(bz-p.gamma_perp*cz/M)-sigE*w.dWz[x,y]
        cayley!(w.ep,w.e,qx,qy,qz,x,y)
        @views w.mp[:,x,y].=w.Mp[x,y].*w.ep[:,x,y]
    end

    field!(w.bh,w.mp,p)
    for y in 1:p.L,x in 1:p.L
        Mm=0.5*(w.M[x,y]+w.Mp[x,y])
        Mm>1e-10 || error("predicted midpoint amplitude approached zero")
        @views w.em[:,x,y].=0.5.*(w.e[:,x,y].+w.ep[:,x,y])
        @views w.mm[:,x,y].=Mm.*w.em[:,x,y]
    end
    field!(w.bs,w.mm,p)

    for y in 1:p.L,x in 1:p.L
        f0=dot(@view(w.e[:,x,y]),@view(w.b0[:,x,y]))
        fp=dot(@view(w.ep[:,x,y]),@view(w.bh[:,x,y]))
        Mn=w.M[x,y]+0.5p.gamma_parallel*p.dt*(f0+fp)+sigM*w.dWp[x,y]
        Mn>1e-10 || error("corrected amplitude approached zero")
        Mm=0.5*(w.M[x,y]+w.Mp[x,y]); ex,ey,ez=w.em[1,x,y],w.em[2,x,y],w.em[3,x,y]
        bx,by,bz=w.bs[1,x,y],w.bs[2,x,y],w.bs[3,x,y]
        cx=ey*bz-ez*by; cy=ez*bx-ex*bz; cz=ex*by-ey*bx
        sigE=sqrt(2p.temperature*p.gamma_perp)/Mm
        qx=p.dt*(bx-p.gamma_perp*cx/Mm)-sigE*w.dWx[x,y]
        qy=p.dt*(by-p.gamma_perp*cy/Mm)-sigE*w.dWy[x,y]
        qz=p.dt*(bz-p.gamma_perp*cz/Mm)-sigE*w.dWz[x,y]
        cayley!(w.en,w.e,qx,qy,qz,x,y)
        @views m[:,x,y].=Mn.*w.en[:,x,y]
    end
end

function initialize(p,kind,seed)
    rng=MersenneTwister(seed); m=zeros(3,p.L,p.L)
    for y in 1:p.L,x in 1:p.L
        v = kind==:random ? randn(rng,3) : [0.12randn(rng),0.12randn(rng),(-1)^(x+y)+0.12randn(rng)]
        v ./= norm(v); m[:,x,y].=0.20.*v
    end
    m
end

mean_amplitude(m,p)=mean(norm(@view m[:,x,y]) for x in 1:p.L,y in 1:p.L)
radial_variance(m,p,M0)=mean((norm(@view m[:,x,y])-M0)^2 for x in 1:p.L,y in 1:p.L)

function harmonic_predictions(p)
    M0=sqrt(p.b+4*(p.J1-p.J2)/p.a)
    E0=(-2p.J1+2p.J2)*M0^2+0.25p.a*(M0^2-p.b)^2
    N=p.L^2
    # Radial Hessian around the uniform-amplitude Neel state.
    invsum=0.0
    Vpp=p.a*(3M0^2-p.b)
    for ny in 0:p.L-1,nx in 0:p.L-1
        qx=2pi*nx/p.L; qy=2pi*ny/p.L
        K=Vpp-2p.J1*(cos(qx)+cos(qy))+4p.J2*cos(qx)*cos(qy)
        invsum+=1/K
    end
    radial_var=p.temperature*invsum/N
    # N radial modes plus 2N-2 nonzero orientational modes.
    energy=E0+(3N-2)*p.temperature/(2N)
    M0,E0,energy,radial_var
end

function simulate(p,kind,seed)
    m=initialize(p,kind,seed); w=W(p.L); rng=MersenneTwister(seed+10_000)
    M0,_,_,_=harmonic_predictions(p)
    ns=div(p.steps,p.sample_every)+1
    t=zeros(ns); E=zeros(ns); A=zeros(ns); R=zeros(ns); ksave=1
    E[1]=energy(m,p)/p.L^2; A[1]=mean_amplitude(m,p); R[1]=radial_variance(m,p,M0)
    for k in 1:p.steps
        strict_step!(m,w,p,rng)
        if k%p.sample_every==0
            ksave+=1; t[ksave]=k*p.dt
            E[ksave]=energy(m,p)/p.L^2; A[ksave]=mean_amplitude(m,p)
            R[ksave]=radial_variance(m,p,M0)
        end
    end
    t,E,A,R
end

function batch_stats(x; nbatch=10)
    nper=div(length(x),nbatch); means=[mean(@view x[(i-1)*nper+1:i*nper]) for i in 1:nbatch]
    mean(x),std(means)/sqrt(nbatch)
end

function smooth(x,n=25)
    [mean(@view x[max(1,i-n+1):i]) for i in eachindex(x)]
end

function main(args=ARGS)
    quick="--quick" in args
    p=quick ? P(L=6,steps=8_000,burn=3_000,sample_every=10) : P()
    tr,Er,Ar,Rr=simulate(p,:random,101)
    to,Eo,Ao,Ro=simulate(p,:neel,202)
    keep=tr .>= p.burn*p.dt
    Erp,Arp,Rrp=Er[keep],Ar[keep],Rr[keep]; Eop,Aop,Rop=Eo[keep],Ao[keep],Ro[keep]
    mer,ser=batch_stats(Erp); meo,seo=batch_stats(Eop)
    mar,sar=batch_stats(Arp); mao,sao=batch_stats(Aop)
    mrr,srr=batch_stats(Rrp); mro,sro=batch_stats(Rop)
    zE=abs(mer-meo)/sqrt(ser^2+seo^2); zA=abs(mar-mao)/sqrt(sar^2+sao^2)
    M0,E0,Eharm,Rharm=harmonic_predictions(p)
    @printf("random start:  <E/N> = %.7f +/- %.2g, <|m|> = %.7f +/- %.2g\n",mer,ser,mar,sar)
    @printf("ordered start: <E/N> = %.7f +/- %.2g, <|m|> = %.7f +/- %.2g\n",meo,seo,mao,sao)
    @printf("between-start differences: z_E = %.3f, z_|m| = %.3f\n",zE,zA)
    @printf("low-T harmonic: E0/N = %.7f, <E/N> = %.7f, M0 = %.7f, <(M-M0)^2> = %.7g\n",E0,Eharm,M0,Rharm)
    @printf("measured radial variance: random %.7g +/- %.2g, ordered %.7g +/- %.2g\n",mrr,srr,mro,sro)

    out=joinpath(@__DIR__,"results"); mkpath(out)
    open(joinpath(out,"j1j2a_finite_temperature_summary.csv"),"w") do io
        println(io,"start,mean_energy_per_site,energy_batch_se,mean_amplitude,amplitude_batch_se,radial_variance,radial_variance_batch_se")
        @printf(io,"random,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",mer,ser,mar,sar,mrr,srr)
        @printf(io,"ordered,%.12g,%.12g,%.12g,%.12g,%.12g,%.12g\n",meo,seo,mao,sao,mro,sro)
        @printf(io,"harmonic,%.12g,NaN,%.12g,NaN,%.12g,NaN\n",Eharm,M0,Rharm)
    end
    PLOTS_AVAILABLE || return
    burntime=p.burn*p.dt
    p1=Plots.plot(tr,smooth(Er),label="random start",xlabel="time",ylabel="energy/site",title="Energy equilibration",framestyle=:box)
    Plots.plot!(p1,to,smooth(Eo),label="Neel start"); Plots.vline!(p1,[burntime],linestyle=:dash,color=:black,label="burn-in")
    Plots.hline!(p1,[Eharm],linestyle=:dot,color=:purple,label="low-T harmonic")
    p2=Plots.plot(tr,smooth(Ar),label="random start",xlabel="time",ylabel="mean |m|",title="Amplitude equilibration",framestyle=:box)
    Plots.plot!(p2,to,smooth(Ao),label="Neel start"); Plots.vline!(p2,[burntime],linestyle=:dash,color=:black,label="burn-in")
    Plots.hline!(p2,[M0],linestyle=:dot,color=:purple,label="T=0 saddle M0")
    p3=Plots.histogram(Erp,bins=35,normalize=:pdf,alpha=.45,label="random start",xlabel="energy/site",ylabel="density",title="Post-burn energy",framestyle=:box)
    Plots.histogram!(p3,Eop,bins=35,normalize=:pdf,alpha=.45,label="Neel start")
    Plots.vline!(p3,[Eharm],linestyle=:dot,color=:purple,label="low-T harmonic")
    p4=Plots.histogram(Rrp,bins=35,normalize=:pdf,alpha=.45,label="random start",xlabel="mean (|m|-M0)^2",ylabel="density",title="Radial-amplitude fluctuations",framestyle=:box)
    Plots.histogram!(p4,Rop,bins=35,normalize=:pdf,alpha=.45,label="Neel start")
    Plots.vline!(p4,[Rharm],linestyle=:dot,color=:purple,label="low-T harmonic")
    title=@sprintf("Finite-T strict Heun+SIB equilibrium: T=%.3g, J2/J1=%.2f",p.temperature,p.J2/p.J1)
    fig=Plots.plot(p1,p2,p3,p4,layout=(2,2),size=(1150,760),margin=5Plots.mm,plot_title=title)
    Plots.savefig(fig,joinpath(out,"j1j2a_finite_temperature.png"))
    Plots.savefig(fig,joinpath(out,"j1j2a_finite_temperature.svg"))
end

main()
