#!/usr/bin/env julia

# Strict coupled longitudinal-Heun / transverse-SIB reference step.
#
# The important distinction is that the two correctors require fields at two
# different configurations:
#
#   Heun amplitude corrector: predicted endpoint m_H = M_p * e_p
#   SIB orientation corrector: predicted midpoint m_S = M_mid * e_mid
#
# Replace `toy_electronic_field` with the constrained electronic solver in a
# production SDW code.  That callback must also readjust the chemical potential
# and return the thermodynamic field b = -dF/dm at every call.

using LinearAlgebra
using Random
using Printf

const Vec3 = NTuple{3,Float64}

@inline add(a::Vec3, b::Vec3)::Vec3 =
    (a[1] + b[1], a[2] + b[2], a[3] + b[3])
@inline sub(a::Vec3, b::Vec3)::Vec3 =
    (a[1] - b[1], a[2] - b[2], a[3] - b[3])
@inline scale(c::Float64, a::Vec3)::Vec3 =
    (c*a[1], c*a[2], c*a[3])
@inline dot3(a::Vec3, b::Vec3) = a[1]*b[1] + a[2]*b[2] + a[3]*b[3]
@inline norm3(a::Vec3) = sqrt(dot3(a, a))
@inline cross3(a::Vec3, b::Vec3)::Vec3 = (
    a[2]*b[3] - a[3]*b[2],
    a[3]*b[1] - a[1]*b[3],
    a[1]*b[2] - a[2]*b[1],
)

# Solve xplus-x = ((x+xplus)/2) cross q analytically.
@inline function cayley(x::Vec3, q::Vec3)::Vec3
    w = scale(0.5, q)
    w2 = dot3(w, w)
    xcrossw = cross3(x, w)
    wx = dot3(w, x)
    den = 1.0 + w2
    return (
        ((1.0-w2)*x[1] + 2.0*xcrossw[1] + 2.0*w[1]*wx)/den,
        ((1.0-w2)*x[2] + 2.0*xcrossw[2] + 2.0*w[2]*wx)/den,
        ((1.0-w2)*x[3] + 2.0*xcrossw[3] + 2.0*w[3]*wx)/den,
    )
end

# Example nonlinear state-dependent thermodynamic field.  A real calculation
# should replace this function with the constrained electronic field solve.
@inline function toy_electronic_field(m::Vec3)::Vec3
    target = (0.55, -0.15, 0.30)
    kappa = 1.1
    quartic = 0.35
    return sub(scale(-kappa, sub(m, target)), scale(quartic*dot3(m, m), m))
end

@inline longitudinal_drift(e::Vec3, b::Vec3, gamma_parallel::Float64) =
    gamma_parallel*dot3(e, b)

# Write the orientational equation as de = e cross a dt + e cross sigma o dW.
@inline function orientation_q(e::Vec3, M::Float64, b::Vec3,
                               h::Float64, dW::Vec3,
                               gamma_perp::Float64, kT::Float64)::Vec3
    M > 1.0e-10 || error("Orientation is undefined for M approximately zero")
    a = sub(b, scale(gamma_perp/M, cross3(e, b)))
    sigma_dW = scale(-sqrt(2.0*kT*gamma_perp)/M, cross3(e, dW))
    return add(scale(h, a), sigma_dW)
end

"""
    strict_step(M, e, h, dW_parallel, dW_perp, field; ...)

Advance one coupled SDW timestep with separate electronic fields for the two
correctors.  The callback `field(m)` is called three times:

1. at the old texture;
2. at the jointly predicted endpoint for Heun;
3. at the predicted midpoint for SIB.

The same Wiener increments are reused in predictor and corrector.
"""
function strict_step(M::Float64, e::Vec3, h::Float64,
                     dW_parallel::Float64, dW_perp::Vec3, field;
                     gamma_parallel::Float64=0.7,
                     gamma_perp::Float64=0.4,
                     kT::Float64=1.0e-3)
    abs(norm3(e) - 1.0) < 1.0e-10 || error("Input orientation must be unit length")
    M > 1.0e-10 || error("M is too small for the orientation equation")

    # Old-state electronic solve.
    m0 = scale(M, e)
    b0 = field(m0)

    # Joint predictor: neither channel is completed before the other.
    sigma_parallel = sqrt(2.0*kT*gamma_parallel)
    fM0 = longitudinal_drift(e, b0, gamma_parallel)
    Mp = M + h*fM0 + sigma_parallel*dW_parallel

    q0 = orientation_q(e, M, b0, h, dW_perp, gamma_perp, kT)
    ep = cayley(e, q0)

    # Heun corrector field: predicted endpoint texture.
    m_endpoint = scale(Mp, ep)
    b_endpoint = field(m_endpoint)
    fMp = longitudinal_drift(ep, b_endpoint, gamma_parallel)
    Mnew = M + 0.5*h*(fM0 + fMp) + sigma_parallel*dW_parallel

    # SIB corrector field: predicted midpoint texture.
    Mmid = 0.5*(M + Mp)
    emid = scale(0.5, add(e, ep))
    m_midpoint = scale(Mmid, emid)
    b_midpoint = field(m_midpoint)
    qmid = orientation_q(emid, Mmid, b_midpoint, h, dW_perp,
                         gamma_perp, kT)
    enew = cayley(e, qmid)

    return Mnew, enew
end

# Cheaper comparison scheme: it reuses the endpoint field in the SIB corrector.
# This is included to make the approximation explicit; it is not the preferred
# state-dependent-field implementation.
function shared_endpoint_step(M::Float64, e::Vec3, h::Float64,
                              dW_parallel::Float64, dW_perp::Vec3, field;
                              gamma_parallel::Float64=0.7,
                              gamma_perp::Float64=0.4,
                              kT::Float64=1.0e-3)
    b0 = field(scale(M, e))
    sigma_parallel = sqrt(2.0*kT*gamma_parallel)
    fM0 = longitudinal_drift(e, b0, gamma_parallel)
    Mp = M + h*fM0 + sigma_parallel*dW_parallel
    ep = cayley(e, orientation_q(e, M, b0, h, dW_perp,
                                 gamma_perp, kT))

    b_endpoint = field(scale(Mp, ep))
    fMp = longitudinal_drift(ep, b_endpoint, gamma_parallel)
    Mnew = M + 0.5*h*(fM0 + fMp) + sigma_parallel*dW_parallel

    Mmid = 0.5*(M + Mp)
    emid = scale(0.5, add(e, ep))
    q_shared = orientation_q(emid, Mmid, b_endpoint, h, dW_perp,
                             gamma_perp, kT)
    enew = cayley(e, q_shared)
    return Mnew, enew
end

function demo(; seed=9417, nsteps=1000, h=1.0e-3, kT=1.0e-3)
    rng = MersenneTwister(seed)
    M1 = 0.45
    e1 = (1.0, 0.0, 0.0)
    M2, e2 = M1, e1

    for _ in 1:nsteps
        dWp = sqrt(h)*randn(rng)
        dWv = (sqrt(h)*randn(rng), sqrt(h)*randn(rng), sqrt(h)*randn(rng))
        M1, e1 = strict_step(M1, e1, h, dWp, dWv, toy_electronic_field; kT=kT)
        M2, e2 = shared_endpoint_step(M2, e2, h, dWp, dWv,
                                      toy_electronic_field; kT=kT)
    end

    m1 = scale(M1, e1)
    m2 = scale(M2, e2)
    @printf("strict final M                 = %.10f\n", M1)
    @printf("strict orientation norm error = %.3e\n", abs(norm3(e1)-1.0))
    @printf("shared-vs-strict |delta m|    = %.3e\n", norm3(sub(m2, m1)))
end

if abspath(PROGRAM_FILE) == @__FILE__
    demo()
end
