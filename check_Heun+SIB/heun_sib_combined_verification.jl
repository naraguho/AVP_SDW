#!/usr/bin/env julia

# Verification of a split longitudinal-Heun + transverse-SIB integrator.
#
# Manufactured model:
#
#   dM = -lambda*(M-Mbar) dt + sigma_M dW_parallel
#
#   de = e cross (-B) dt
#        - sqrt(2D) e cross o dW_vector,       |e|=1
#
#   m = M*e.
#
# The amplitude is an additive-noise Ornstein-Uhlenbeck process.  The direction
# is Stratonovich rotational diffusion integrated by genuine SIB/Cayley.
# The channels are independent, which gives exact means and moments for weak
# tests while still exposing the order that limits the complete vector method.
#
# Expected results:
#
#   deterministic:  M, e, and m all have global order 2
#   stochastic strong: M order 1, e order 1/2, m order 1/2
#   stochastic weak:   amplitude moments order 2, e and m order 1
#
# This is a verification of the numerical composition.  It does not prove that
# a particular SDW local-field solver or fluctuation-dissipation law is correct.
#
# Usage:
#   julia heun_sib_combined_verification.jl
#   julia heun_sib_combined_verification.jl --quick

using Random
using LinearAlgebra
using Printf
using DelimitedFiles

const PLOTS_AVAILABLE = try
    @eval import Plots
    true
catch
    false
end

const Vec3 = NTuple{3,Float64}

@inline add(a::Vec3,b::Vec3)::Vec3 = (a[1]+b[1],a[2]+b[2],a[3]+b[3])
@inline sub(a::Vec3,b::Vec3)::Vec3 = (a[1]-b[1],a[2]-b[2],a[3]-b[3])
@inline scale(c::Float64,a::Vec3)::Vec3 = (c*a[1],c*a[2],c*a[3])
@inline dot3(a::Vec3,b::Vec3) = a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
@inline norm3(a::Vec3) = sqrt(dot3(a,a))
@inline cross3(a::Vec3,b::Vec3)::Vec3 = (
    a[2]*b[3]-a[3]*b[2],
    a[3]*b[1]-a[1]*b[3],
    a[1]*b[2]-a[2]*b[1],
)

@inline function cayley(e::Vec3,q::Vec3)::Vec3
    w = scale(0.5,q)
    w2 = dot3(w,w)
    c = cross3(e,w)
    d = dot3(w,e)
    den = 1.0+w2
    return (
        ((1.0-w2)*e[1]+2.0*c[1]+2.0*w[1]*d)/den,
        ((1.0-w2)*e[2]+2.0*c[2]+2.0*w[2]*d)/den,
        ((1.0-w2)*e[3]+2.0*c[3]+2.0*w[3]*d)/den,
    )
end

@inline function amplitude_heun(M::Float64,h::Float64,dW::Float64,
                                lambda::Float64,Mbar::Float64,
                                sigma_M::Float64)
    f0 = -lambda*(M-Mbar)
    Mp = M+h*f0+sigma_M*dW
    fp = -lambda*(Mp-Mbar)
    return M+0.5*h*(f0+fp)+sigma_M*dW
end

# For alpha=0 and a constant field, sigma is independent of e.  Predictor and
# corrector are nevertheless written explicitly to retain the SIB structure.
@inline function direction_sib(e::Vec3,h::Float64,dW::Vec3,
                               B::Vec3,D::Float64)::Vec3
    noise = scale(-sqrt(2.0*D),dW)
    q0 = add(scale(-h,B),noise)
    ep = cayley(e,q0)
    ehalf = scale(0.5,add(e,ep))
    # a(ehalf)=-B and sigma(ehalf)dW=-sqrt(2D)dW in this benchmark.
    q1 = add(scale(-h,B),noise)
    return cayley(e,q1)
end

@inline function hybrid_step(M,e,h,dW_M,dW_e,lambda,Mbar,sigma_M,B,D)
    Mnew = amplitude_heun(M,h,dW_M,lambda,Mbar,sigma_M)
    enew = direction_sib(e,h,dW_e,B,D)
    return Mnew,enew
end

function fit_slope(h,error)
    x=log.(h); y=log.(error)
    return (hcat(ones(length(x)),x)\y)[2]
end

function deterministic_test(output_dir)
    T=1.0; lambda=0.7; Mbar=1.0; Minit=0.8
    e0=(1.0,0.0,0.0); B=(0.0,0.0,1.0)
    Ns=2 .^ collect(2:8); hs=T./Ns
    errM=Float64[]; erre=Float64[]; errm=Float64[]
    Mexact=Mbar+(Minit-Mbar)*exp(-lambda*T)
    eexact=(cos(T),sin(T),0.0)
    mexact=scale(Mexact,eexact)
    for (N,h) in zip(Ns,hs)
        M=Minit; e=e0
        for _ in 1:N
            M,e=hybrid_step(M,e,h,0.0,(0.0,0.0,0.0),
                            lambda,Mbar,0.0,B,0.0)
        end
        push!(errM,abs(M-Mexact))
        push!(erre,norm3(sub(e,eexact)))
        push!(errm,norm3(sub(scale(M,e),mexact)))
    end
    open(joinpath(output_dir,"deterministic_combined_convergence.csv"),"w") do io
        println(io,"h,amplitude_error,direction_error,total_vector_error")
        writedlm(io,hcat(hs,errM,erre,errm),',')
    end
    return hs,errM,erre,errm,fit_slope(hs,errM),fit_slope(hs,erre),fit_slope(hs,errm)
end

function aggregate_scalar(a,first,block)
    s=0.0
    @inbounds for j in first:first+block-1
        s+=a[j]
    end
    return s
end

function aggregate_vec(a,first,block)::Vec3
    x=0.0; y=0.0; z=0.0
    @inbounds for j in first:first+block-1
        x+=a[1,j]; y+=a[2,j]; z+=a[3,j]
    end
    return (x,y,z)
end

function strong_test(output_dir;npaths=4000)
    rng=MersenneTwister(81472)
    T=1.0; lambda=0.7; Mbar=1.0; Minit=0.8; sigma_M=0.08
    e0=(1.0,0.0,0.0); B=(0.0,0.0,1.0); D=0.03
    exponents=collect(5:9); ref_exponent=13
    Nref=2^ref_exponent; href=T/Nref
    hs=T./(2 .^ exponents)
    sumM=zeros(length(hs)); sume=zeros(length(hs)); summ=zeros(length(hs))
    dWM=zeros(Nref); dWe=zeros(3,Nref)

    for _ in 1:npaths
        randn!(rng,dWM); dWM .*= sqrt(href)
        randn!(rng,dWe); dWe .*= sqrt(href)
        Mr=Minit; er=e0
        @inbounds for j in 1:Nref
            Mr,er=hybrid_step(Mr,er,href,dWM[j],
                (dWe[1,j],dWe[2,j],dWe[3,j]),lambda,Mbar,sigma_M,B,D)
        end
        mr=scale(Mr,er)

        for (k,exponent) in enumerate(exponents)
            N=2^exponent; h=T/N; block=div(Nref,N)
            M=Minit; e=e0; first=1
            for _ in 1:N
                dwm=aggregate_scalar(dWM,first,block)
                dwe=aggregate_vec(dWe,first,block)
                M,e=hybrid_step(M,e,h,dwm,dwe,lambda,Mbar,sigma_M,B,D)
                first+=block
            end
            sumM[k]+=(M-Mr)^2
            sume[k]+=dot3(sub(e,er),sub(e,er))
            dm=sub(scale(M,e),mr)
            summ[k]+=dot3(dm,dm)
        end
    end
    rmsM=sqrt.(sumM./npaths); rmse=sqrt.(sume./npaths); rmsm=sqrt.(summ./npaths)
    open(joinpath(output_dir,"stochastic_strong_combined_convergence.csv"),"w") do io
        println(io,"h,amplitude_RMS_error,direction_RMS_error,total_vector_RMS_error")
        writedlm(io,hcat(hs,rmsM,rmse,rmsm),',')
    end
    return hs,rmsM,rmse,rmsm,fit_slope(hs,rmsM),fit_slope(hs,rmse),fit_slope(hs,rmsm)
end

function simpson(f,a,b,n)
    iseven(n)||error("n must be even")
    dx=(b-a)/n; s=f(a)+f(b)
    for i in 1:n-1
        s+=(isodd(i) ? 4.0 : 2.0)*f(a+i*dx)
    end
    return s*dx/3.0
end

function direction_mean_factor(h,D)
    lambda=0.5*D*h
    integrand(z)=begin
        r2=lambda*z^2
        ((1.0-r2/3.0)/(1.0+r2))*sqrt(2.0/pi)*z^2*exp(-0.5*z^2)
    end
    return simpson(integrand,0.0,10.0,40_000)
end

function weak_test(output_dir)
    T=1.0; lambda=0.7; Mbar=1.0; Minit=0.8; sigma_M=0.08; D=0.10
    Ns=2 .^ collect(2:10); hs=T./Ns
    exact_mean_M=Mbar+(Minit-Mbar)*exp(-lambda*T)
    exact_var_M=sigma_M^2/(2.0*lambda)*(1.0-exp(-2.0*lambda*T))
    exact_M2=exact_var_M+exact_mean_M^2
    exact_mean_e=exp(-2.0*D*T)
    exact_mean_m=exact_mean_M*exact_mean_e
    error_M=Float64[]; error_M2=Float64[]; error_e=Float64[]; error_m=Float64[]

    for (N,h) in zip(Ns,hs)
        A=1.0-lambda*h+0.5*(lambda*h)^2
        noise_coefficient=sigma_M*(1.0-0.5*lambda*h)
        meanM=Mbar+A^N*(Minit-Mbar)
        varM=noise_coefficient^2*h*(1.0-A^(2N))/(1.0-A^2)
        meanM2=varM+meanM^2
        c=direction_mean_factor(h,D)
        meane=c^N
        meanm=meanM*meane
        push!(error_M,abs(meanM-exact_mean_M))
        push!(error_M2,abs(meanM2-exact_M2))
        push!(error_e,abs(meane-exact_mean_e))
        push!(error_m,abs(meanm-exact_mean_m))
    end
    open(joinpath(output_dir,"stochastic_weak_combined_convergence.csv"),"w") do io
        println(io,"h,error_mean_M,error_mean_M2,error_mean_ex,error_mean_mx")
        writedlm(io,hcat(hs,error_M,error_M2,error_e,error_m),',')
    end
    return hs,error_M,error_M2,error_e,error_m,
           fit_slope(hs,error_M),fit_slope(hs,error_M2),
           fit_slope(hs,error_e),fit_slope(hs,error_m)
end

function geometry_test()
    rng=MersenneTwister(92); h=0.02
    M=0.8; e=(1.0,0.0,0.0); max_e_norm=0.0; max_factorization=0.0
    for _ in 1:100_000
        dWM=sqrt(h)*randn(rng)
        dWe=(sqrt(h)*randn(rng),sqrt(h)*randn(rng),sqrt(h)*randn(rng))
        M,e=hybrid_step(M,e,h,dWM,dWe,0.7,1.0,0.02,(0.0,0.0,1.0),0.01)
        max_e_norm=max(max_e_norm,abs(norm3(e)-1.0))
        max_factorization=max(max_factorization,abs(norm3(scale(M,e))-abs(M)))
    end
    return max_e_norm,max_factorization
end

function make_plot(output_dir,dh,dM,de,dm,sh,sM,se,sm,wh,wM2,we,wm)
    PLOTS_AVAILABLE||return
    p1=Plots.plot(dh,dM,xscale=:log10,yscale=:log10,marker=:circle,label="M",
                  xlabel="h",ylabel="error",title="deterministic convergence")
    Plots.plot!(p1,dh,de,marker=:circle,label="e")
    Plots.plot!(p1,dh,dm,marker=:circle,label="m=M e")
    guide=dm[end].*(dh./dh[end]).^2
    Plots.plot!(p1,dh,guide,linestyle=:dash,label="slope 2")

    p2=Plots.plot(sh,sM,xscale=:log10,yscale=:log10,marker=:circle,label="M",
                  xlabel="h",ylabel="RMS error",title="strong convergence")
    Plots.plot!(p2,sh,se,marker=:circle,label="e")
    Plots.plot!(p2,sh,sm,marker=:circle,label="m=M e")
    guidehalf=sm[end].*(sh./sh[end]).^0.5
    Plots.plot!(p2,sh,guidehalf,linestyle=:dash,label="slope 1/2")

    p3=Plots.plot(wh,wM2,xscale=:log10,yscale=:log10,marker=:circle,label="E[M^2]",
                  xlabel="h",ylabel="weak error",title="weak convergence")
    Plots.plot!(p3,wh,we,marker=:circle,label="E[e_x]")
    Plots.plot!(p3,wh,wm,marker=:circle,label="E[m_x]")
    guide1=wm[end].*(wh./wh[end])
    Plots.plot!(p3,wh,guide1,linestyle=:dash,label="slope 1")

    p4=Plots.plot(axis=false,ticks=false,legend=false,title="expected limiting orders")
    Plots.annotate!(p4,0.05,0.72,Plots.text("deterministic full vector: 2",14,:left))
    Plots.annotate!(p4,0.05,0.50,Plots.text("strong full vector: 1/2",14,:left))
    Plots.annotate!(p4,0.05,0.28,Plots.text("weak full vector: 1",14,:left))
    fig=Plots.plot(p1,p2,p3,p4,layout=(2,2),size=(1200,900),margin=3Plots.mm)
    Plots.savefig(fig,joinpath(output_dir,"heun_sib_combined_verification.png"))
end

function main(args=ARGS)
    quick="--quick" in args
    output_dir=quick ? "heun_sib_combined_results_quick" : "heun_sib_combined_results"
    mkpath(output_dir)
    npaths=quick ? 500 : 4000

    dh,dM,de,dm,pdM,pde,pdm=deterministic_test(output_dir)
    sh,sM,se,sm,psM,pse,psm=strong_test(output_dir;npaths=npaths)
    wh,wM,wM2,we,wm,pwM,pwM2,pwe,pwm=weak_test(output_dir)
    normerr,factorerr=geometry_test()
    make_plot(output_dir,dh,dM,de,dm,sh,sM,se,sm,wh,wM2,we,wm)

    path=joinpath(output_dir,"summary.txt")
    open(path,"w") do io
        println(io,"Combined longitudinal-Heun + transverse-SIB verification")
        @printf(io,"deterministic amplitude order = %.6f (expected 2)\n",pdM)
        @printf(io,"deterministic direction order = %.6f (expected 2)\n",pde)
        @printf(io,"deterministic total-vector order = %.6f (expected 2)\n",pdm)
        @printf(io,"strong amplitude order = %.6f (expected about 1)\n",psM)
        @printf(io,"strong direction order = %.6f (expected 0.5)\n",pse)
        @printf(io,"strong total-vector order = %.6f (expected 0.5)\n",psm)
        @printf(io,"weak amplitude mean order = %.6f (expected 2)\n",pwM)
        @printf(io,"weak amplitude second-moment order = %.6f (expected 2)\n",pwM2)
        @printf(io,"weak direction mean order = %.6f (expected 1)\n",pwe)
        @printf(io,"weak total-vector mean order = %.6f (expected 1)\n",pwm)
        @printf(io,"max direction-norm error = %.6e\n",normerr)
        @printf(io,"max |norm(M e)-|M|| error = %.6e\n",factorerr)
    end
    println(read(path,String))
    println("Results: ",abspath(output_dir))
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end

