#!/usr/bin/env julia

# Standalone CPU verification of Semi-Implicit Scheme B (SIB) for the
# stochastic Landau-Lifshitz equation in Mentink et al., JPCM 22, 176001
# (2010), arXiv:1002.1801.
#
# This file tests four distinct claims.  They should not be conflated:
#
# 1. Geometry: every Cayley/SIB step preserves |X| to floating-point roundoff.
# 2. Deterministic accuracy: when D=0, the global error is second order in h.
# 3. Stochastic accuracy: for noncommutative multiplicative noise, the expected
#    mean-square/strong order is 1/2 and the weak order is 1.
# 4. Exchange conservation: for two undamped Heisenberg spins, simultaneous
#    SIB updates preserve total spin and exchange energy to roundoff.
#
# It also records deterministic phase-error growth.  For a unit spin in a
# constant unit field, Cayley precession advances by
#
#     theta_h = 2 atan(h/2)
#
# per step instead of h.  Hence the small-h phase error after time t is
# approximately
#
#     delta_phi(t) = t*h^2/12.
#
# Usage:
#   julia sib_integrator_verification.jl
#   julia sib_integrator_verification.jl --quick
#
# Outputs are written to sib_verification_results/.

using Random
using LinearAlgebra
using Statistics
using Printf
using DelimitedFiles

const PLOTS_AVAILABLE = try
    @eval import Plots
    true
catch
    false
end

const Spin = NTuple{3,Float64}

@inline add(a::Spin,b::Spin)::Spin = (a[1]+b[1],a[2]+b[2],a[3]+b[3])
@inline sub(a::Spin,b::Spin)::Spin = (a[1]-b[1],a[2]-b[2],a[3]-b[3])
@inline scale(c::Float64,a::Spin)::Spin = (c*a[1],c*a[2],c*a[3])
@inline dot3(a::Spin,b::Spin) = a[1]*b[1]+a[2]*b[2]+a[3]*b[3]
@inline norm3(a::Spin) = sqrt(dot3(a,a))
@inline cross3(a::Spin,b::Spin)::Spin = (
    a[2]*b[3]-a[3]*b[2],
    a[3]*b[1]-a[1]*b[3],
    a[1]*b[2]-a[2]*b[1],
)

# Exact 3D solve of
#   Xnew-Xold = ((Xold+Xnew)/2) cross q.
@inline function cayley(X::Spin,q::Spin)::Spin
    w = scale(0.5,q)
    w2 = dot3(w,w)
    c = cross3(X,w)
    d = dot3(w,X)
    den = 1.0+w2
    return (
        ((1.0-w2)*X[1]+2.0*c[1]+2.0*w[1]*d)/den,
        ((1.0-w2)*X[2]+2.0*c[2]+2.0*w[2]*d)/den,
        ((1.0-w2)*X[3]+2.0*c[3]+2.0*w[3]*d)/den,
    )
end

# Paper notation:
#   dX = X cross a(X) dt + X cross sigma(X) o dW
#   a(X) = -B - alpha X cross B
#   sigma(X)dW = -sqrt(2D)dW - alpha sqrt(2D) X cross dW.
@inline function drift_a(X::Spin,B::Spin,alpha::Float64)::Spin
    return sub(scale(-1.0,B),scale(alpha,cross3(X,B)))
end

@inline function sigma_dW(X::Spin,dW::Spin,alpha::Float64,D::Float64)::Spin
    s = sqrt(2.0*D)
    return sub(scale(-s,dW),scale(alpha*s,cross3(X,dW)))
end

# Genuine two-iteration SIB.  The same Brownian increment is used in both
# predictor and corrector.
@inline function sib_step_constant_field(X::Spin,h::Float64,dW::Spin,
                                         B::Spin,alpha::Float64,D::Float64)::Spin
    q0 = add(scale(h,drift_a(X,B,alpha)),sigma_dW(X,dW,alpha,D))
    Y = cayley(X,q0)
    Xhalf = scale(0.5,add(X,Y))
    q1 = add(scale(h,drift_a(Xhalf,B,alpha)),sigma_dW(Xhalf,dW,alpha,D))
    return cayley(X,q1)
end

function evolve_constant_field(X0::Spin,h::Float64,dW_steps,
                               B::Spin,alpha::Float64,D::Float64)::Spin
    X = X0
    for dW in dW_steps
        X = sib_step_constant_field(X,h,dW,B,alpha,D)
    end
    return X
end

function linear_fit_slope(h,error)
    mask = [isfinite(error[i]) && error[i]>0.0 for i in eachindex(error)]
    x = log.(h[mask]); y = log.(error[mask])
    A = hcat(ones(length(x)),x)
    coeff = A\y
    return coeff[2]
end

# Composite Simpson integration on [a,b].
function simpson_integral(f,a::Float64,b::Float64,n::Int)
    iseven(n) || error("Simpson panel count must be even")
    dx = (b-a)/n
    total = f(a)+f(b)
    for i in 1:n-1
        total += (isodd(i) ? 4.0 : 2.0)*f(a+i*dx)
    end
    return total*dx/3.0
end

# Exact weak benchmark for isotropic rotational Brownian motion:
#
#   dX = -sqrt(2D) X cross o dW,       |X(0)|=1.
#
# The exact first moment is E[X(T)] = exp(-2DT)X(0).  For one SIB/Cayley
# step, isotropy gives E[X_{n+1}|X_n] = c(h)X_n, where
#
#   c(h)=E[(1-r^2/3)/(1+r^2)],
#   r^2=(D h/2) chi^2_3.
#
# We evaluate this one-dimensional radial expectation by deterministic
# quadrature, so the weak-order slope is not obscured by Monte Carlo noise.
function exact_weak_rotational_diffusion(output_dir::String)
    D = 0.10
    T = 1.0
    levels = 2 .^ collect(2:10)
    hs = T ./ levels
    exact_mean = exp(-2.0*D*T)
    numerical_mean = Float64[]
    errors = Float64[]

    for (N,h) in zip(levels,hs)
        lambda = 0.5*D*h
        integrand(z) = begin
            r2 = lambda*z^2
            cayley_factor = (1.0-r2/3.0)/(1.0+r2)
            maxwell_density = sqrt(2.0/pi)*z^2*exp(-0.5*z^2)
            cayley_factor*maxwell_density
        end
        c = simpson_integral(integrand,0.0,10.0,40_000)
        mean_h = c^N
        push!(numerical_mean,mean_h)
        push!(errors,abs(mean_h-exact_mean))
    end
    slope = linear_fit_slope(hs,errors)
    open(joinpath(output_dir,"exact_weak_convergence.csv"),"w") do io
        println(io,"h,numerical_mean_Xx,exact_mean_Xx,absolute_weak_error")
        writedlm(io,hcat(hs,numerical_mean,fill(exact_mean,length(hs)),errors),',')
    end
    return hs,errors,slope
end

function deterministic_convergence(output_dir::String)
    T = 1.0
    B = (0.0,0.0,1.0)
    X0 = (1.0,0.0,0.0)
    levels = 2 .^ collect(2:8)
    hs = T ./ levels
    errors = Float64[]
    Xexact = (cos(T),sin(T),0.0)
    for (N,h) in zip(levels,hs)
        X = X0
        zero_dW = (0.0,0.0,0.0)
        for _ in 1:N
            X = sib_step_constant_field(X,h,zero_dW,B,0.0,0.0)
        end
        push!(errors,norm3(sub(X,Xexact)))
    end
    slope = linear_fit_slope(hs,errors)
    writedlm(joinpath(output_dir,"deterministic_convergence.csv"),
             hcat(hs,errors),',')
    return hs,errors,slope
end

function deterministic_error_growth(output_dir::String)
    h = 0.05
    T = 20.0
    N = round(Int,T/h)
    B = (0.0,0.0,1.0)
    X = (1.0,0.0,0.0)
    rows = zeros(Float64,N+1,4)
    theta_h = 2.0*atan(h/2.0)
    for n in 0:N
        t = n*h
        Xexact = (cos(t),sin(t),0.0)
        rows[n+1,1] = t
        rows[n+1,2] = norm3(sub(X,Xexact))
        rows[n+1,3] = abs(n*(h-theta_h))
        rows[n+1,4] = t*h^2/12.0
        n<N && (X = sib_step_constant_field(X,h,(0.0,0.0,0.0),B,0.0,0.0))
    end
    open(joinpath(output_dir,"deterministic_error_growth.csv"),"w") do io
        println(io,"time,vector_error,exact_cayley_phase_error,leading_t_h2_over_12")
        writedlm(io,rows,',')
    end
    return rows
end

function aggregate_increment(dWfine,first_index::Int,block::Int)::Spin
    sx=0.0; sy=0.0; sz=0.0
    last_index = first_index+block-1
    @inbounds for j in first_index:last_index
        sx += dWfine[1,j]
        sy += dWfine[2,j]
        sz += dWfine[3,j]
    end
    return (sx,sy,sz)
end

function stochastic_convergence(output_dir::String;npaths::Int=4000)
    rng = MersenneTwister(73021)
    T = 1.0
    B = (0.0,0.0,1.0)
    X0 = (1.0,0.0,0.0)
    alpha = 0.15
    D = 0.03
    # Denser strong-convergence grid.  Nrefs is chosen so every coarse N
    # divides it exactly, preserving exact Brownian coupling.
    Ns = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]
    Nrefs = 3 * 2^15   # 98304
    all(Nrefs % N == 0 for N in Ns) || error("Every coarse N must divide Nrefs")
    href = T/Nrefs
    hs = T ./ Ns
    strong_sums = zeros(length(hs))
    weak_sums = zeros(length(hs))
    weak_sq_sums = zeros(length(hs))

    dWfine = zeros(Float64,3,Nrefs)
    for _ in 1:npaths
        randn!(rng,dWfine)
        dWfine .*= sqrt(href)

        Xref = X0
        @inbounds for j in 1:Nrefs
            dW = (dWfine[1,j],dWfine[2,j],dWfine[3,j])
            Xref = sib_step_constant_field(Xref,href,dW,B,alpha,D)
        end

        for (ell,N) in enumerate(Ns)
            h = T/N
            block = div(Nrefs,N)
            X = X0
            first_index = 1
            for _ in 1:N
                dW = aggregate_increment(dWfine,first_index,block)
                X = sib_step_constant_field(X,h,dW,B,alpha,D)
                first_index += block
            end
            difference = sub(X,Xref)
            strong_sums[ell] += dot3(difference,difference)
            # Smooth observable for weak convergence: phi(X)=X_z.
            paired = X[3]-Xref[3]
            weak_sums[ell] += paired
            weak_sq_sums[ell] += paired^2
        end
    end

    strong = sqrt.(strong_sums ./ npaths)
    weak_signed = weak_sums ./ npaths
    weak_abs = abs.(weak_signed)
    weak_var = max.(weak_sq_sums./npaths .- weak_signed.^2,0.0)
    weak_se = sqrt.(weak_var./npaths)
    strong_slope = linear_fit_slope(hs,strong)

    # Pairwise/local effective strong order between neighboring h values.
    strong_peff = [
        log(strong[i]/strong[i+1]) / log(hs[i]/hs[i+1])
        for i in 1:length(hs)-1
    ]

    # For E_strong ~ C*sqrt(h), this should approach a constant plateau.
    strong_scaled = strong ./ sqrt.(hs)

    # Fine-grid fit using the five smallest h values.
    nfine = min(5,length(hs))
    fine_idx = (length(hs)-nfine+1):length(hs)
    strong_slope_fine = linear_fit_slope(hs[fine_idx],strong[fine_idx])

    # Fit weak order only where the estimated bias is resolved above its
    # Monte Carlo standard error.  This avoids fitting pure sampling noise.
    resolved = weak_abs .> 2.0.*weak_se
    weak_slope = count(resolved)>=3 ? linear_fit_slope(hs[resolved],weak_abs[resolved]) : NaN

    open(joinpath(output_dir,"stochastic_convergence.csv"),"w") do io
        println(io,"N,h,strong_rms_error,strong_over_sqrt_h,weak_signed_error_Xz,weak_abs_error_Xz,weak_standard_error")
        writedlm(io,hcat(Ns,hs,strong,strong_scaled,weak_signed,weak_abs,weak_se),',')
    end

    open(joinpath(output_dir,"strong_effective_order.csv"),"w") do io
        println(io,"h_left,h_right,p_effective")
        writedlm(io,hcat(hs[1:end-1],hs[2:end],strong_peff),',')
    end

    return hs,strong,strong_scaled,strong_peff,weak_abs,weak_se,strong_slope,strong_slope_fine,weak_slope
end

# Simultaneous two-spin SIB for pure Heisenberg exchange and alpha=D=0.
function sib_two_spin_step(X1::Spin,X2::Spin,h::Float64,J::Float64)
    a1 = scale(-J,X2)
    a2 = scale(-J,X1)
    Y1 = cayley(X1,scale(h,a1))
    Y2 = cayley(X2,scale(h,a2))
    Z1 = scale(0.5,add(X1,Y1))
    Z2 = scale(0.5,add(X2,Y2))
    X1new = cayley(X1,scale(-h*J,Z2))
    X2new = cayley(X2,scale(-h*J,Z1))
    return X1new,X2new
end

function conservation_tests(output_dir::String)
    # Long stochastic trajectory: individual norm preservation.
    rng = MersenneTwister(917)
    X = (1.0,0.0,0.0)
    h = 0.05
    B = (0.2,-0.1,1.0)
    max_norm_error = 0.0
    for _ in 1:100_000
        dW = (sqrt(h)*randn(rng),sqrt(h)*randn(rng),sqrt(h)*randn(rng))
        X = sib_step_constant_field(X,h,dW,B,0.1,0.02)
        max_norm_error = max(max_norm_error,abs(norm3(X)-1.0))
    end

    # Undamped two-spin exchange: total spin and energy preservation.
    X1 = (1.0,0.0,0.0)
    X2raw = (-0.5,sqrt(3.0)/2.0,0.0)
    X2 = scale(1.0/norm3(X2raw),X2raw)
    J = 1.0
    total0 = add(X1,X2)
    energy0 = -J*dot3(X1,X2)
    max_total_error = 0.0
    max_energy_error = 0.0
    for _ in 1:100_000
        X1,X2 = sib_two_spin_step(X1,X2,0.10,J)
        max_total_error = max(max_total_error,norm3(sub(add(X1,X2),total0)))
        max_energy_error = max(max_energy_error,abs(-J*dot3(X1,X2)-energy0))
    end

    open(joinpath(output_dir,"conservation.txt"),"w") do io
        @printf(io,"max_single_spin_norm_error = %.16e\n",max_norm_error)
        @printf(io,"max_two_spin_total_spin_error = %.16e\n",max_total_error)
        @printf(io,"max_two_spin_exchange_energy_error = %.16e\n",max_energy_error)
    end
    return max_norm_error,max_total_error,max_energy_error
end


function make_strong_diagnostic_plots(output_dir,hs,strong,strong_scaled,strong_peff)
    PLOTS_AVAILABLE || return nothing

    p1 = Plots.plot(hs,strong,xscale=:log10,yscale=:log10,marker=:circle,
                    xlabel="h",ylabel="RMS path error",label="strong error",
                    title="dense stochastic strong convergence",legend=:topleft)
    guidehalf = strong[end].*(hs./hs[end]).^0.5
    Plots.plot!(p1,hs,guidehalf,linestyle=:dash,label="slope 1/2")
    Plots.savefig(p1,joinpath(output_dir,"strong_convergence_dense.png"))

    p2 = Plots.plot(hs,strong_scaled,xscale=:log10,marker=:circle,
                    xlabel="h",ylabel="E_strong / sqrt(h)",label="scaled error",
                    title="strong-error plateau diagnostic",legend=:best)
    Plots.savefig(p2,joinpath(output_dir,"strong_scaled_plateau.png"))

    hmid = sqrt.(hs[1:end-1].*hs[2:end])
    p3 = Plots.plot(hmid,strong_peff,xscale=:log10,marker=:circle,
                    xlabel="geometric-mean h",ylabel="local effective order",
                    label="p_eff",title="local strong convergence order",legend=:best)
    Plots.hline!(p3,[0.5],linestyle=:dash,label="expected 1/2")
    Plots.savefig(p3,joinpath(output_dir,"strong_effective_order.png"))
    return nothing
end

function make_plots(output_dir,det_h,det_err,st_h,strong,weak_h,weak,growth)
    PLOTS_AVAILABLE || return nothing
    p1 = Plots.plot(det_h,det_err,xscale=:log10,yscale=:log10,marker=:circle,
                    xlabel="h",ylabel="endpoint error",label="SIB",
                    title="deterministic order 2",legend=:topleft)
    guide2 = det_err[end].*(det_h./det_h[end]).^2
    Plots.plot!(p1,det_h,guide2,linestyle=:dash,label="slope 2")

    p2 = Plots.plot(st_h,strong,xscale=:log10,yscale=:log10,marker=:circle,
                    xlabel="h",ylabel="RMS path error",label="strong error",
                    title="stochastic strong order 1/2",legend=:topleft)
    guidehalf = strong[end].*(st_h./st_h[end]).^0.5
    Plots.plot!(p2,st_h,guidehalf,linestyle=:dash,label="slope 1/2")

    p3 = Plots.plot(weak_h,weak,xscale=:log10,yscale=:log10,
                    marker=:circle,xlabel="h",ylabel="weak error in E[X_x]",
                    label="exact moment benchmark",
                    title="stochastic weak order 1",legend=:topleft)
    guide1 = weak[end].*(weak_h./weak_h[end])
    Plots.plot!(p3,weak_h,guide1,linestyle=:dash,label="slope 1")

    p4 = Plots.plot(growth[:,1],growth[:,2],label="vector error",linewidth=2,
                    xlabel="time",ylabel="error",title="Cayley phase-error growth")
    Plots.plot!(p4,growth[:,1],growth[:,3],linestyle=:dash,
                label="exact phase difference")
    Plots.plot!(p4,growth[:,1],growth[:,4],linestyle=:dot,
                label="t h^2 / 12")
    fig = Plots.plot(p1,p2,p3,p4,layout=(2,2),size=(1200,900),margin=3Plots.mm)
    Plots.savefig(fig,joinpath(output_dir,"sib_verification.png"))
    return nothing
end

function main(args=ARGS)
    quick = "--quick" in args
    output_dir = quick ? "sib_verification_improved_results_quick" : "sib_verification_improved_results"
    mkpath(output_dir)
    npaths = quick ? 500 : 4000

    println("Running deterministic convergence test...")
    det_h,det_err,det_slope = deterministic_convergence(output_dir)
    println("Running deterministic error-growth test...")
    growth = deterministic_error_growth(output_dir)
    println("Running stochastic coupled-path convergence test with $npaths paths...")
    st_h,strong,strong_scaled,strong_peff,paired_weak,paired_weak_se,strong_slope,strong_slope_fine,paired_weak_slope =
        stochastic_convergence(output_dir;npaths=npaths)
    println("Running exact-moment weak convergence test...")
    weak_h,weak,weak_slope = exact_weak_rotational_diffusion(output_dir)
    println("Running long conservation tests...")
    norm_error,total_error,energy_error = conservation_tests(output_dir)
    make_plots(output_dir,det_h,det_err,st_h,strong,weak_h,weak,growth)
    make_strong_diagnostic_plots(output_dir,st_h,strong,strong_scaled,strong_peff)

    summary_path = joinpath(output_dir,"summary.txt")
    open(summary_path,"w") do io
        println(io,"SIB verification against expected analytical/numerical properties")
        @printf(io,"deterministic fitted order = %.6f (expected 2)\n",det_slope)
        @printf(io,"stochastic strong global fitted order = %.6f (expected 0.5)\n",strong_slope)
        @printf(io,"stochastic strong fine-grid fitted order = %.6f (five smallest h; expected 0.5)\n",strong_slope_fine)
        @printf(io,"stochastic weak fitted order = %.6f (expected 1; exact-moment benchmark)\n",weak_slope)
        @printf(io,"paired Monte Carlo weak fitted order = %.6f (diagnostic only)\n",paired_weak_slope)
        @printf(io,"max norm error over 100000 stochastic steps = %.6e\n",norm_error)
        @printf(io,"max two-spin total-spin error over 100000 steps = %.6e\n",total_error)
        @printf(io,"max two-spin energy error over 100000 steps = %.6e\n",energy_error)
        println(io,"Important: exact norm/conservation does not imply zero trajectory error.")
    end

    println()
    println(read(summary_path,String))
    println("Pairwise strong effective orders:")
    for i in eachindex(strong_peff)
        @printf("  h = %.8f -> %.8f : p_eff = %.6f\n",st_h[i],st_h[i+1],strong_peff[i])
    end
    println("Results: ",abspath(output_dir))
end

if abspath(PROGRAM_FILE)==@__FILE__
    main()
end
