# AVP/SDW stochastic dynamics: integrator notes and verification

This repository documents the numerical time integrator used for a soft spin-density-wave (SDW) order parameter.  The order-parameter amplitude and orientation obey different dynamics, so they are advanced with different methods:

- the longitudinal amplitude is advanced by a stochastic Heun predictor-corrector;
- the transverse orientation is advanced by Semi-Implicit Scheme B (SIB), which retains an implicit-midpoint rotation at both SIB stages.

The material is divided by purpose:

- [`integrator/`](integrator/): physical equation, longitudinal/transverse decomposition, SIB citation, and the production implementation protocol;
- [`check_SIB/`](check_SIB/): isolated CPU verification of the transverse SIB method;
- [`check_Heun+SIB/`](check_Heun+SIB/): manufactured-problem verification of the complete longitudinal-Heun/transverse-SIB composition.

These checks establish the expected numerical orders and geometric preservation of the integrator. They do **not** by themselves validate the electronic free-energy calculation, the constrained local-field solver, or thermodynamic parameters in a particular SDW simulation. Those parts require an additional refinement study using the actual SDW field routine.

## Quick start

From the repository root:

```bash
julia check_SIB/sib_integrator_verification.jl --quick
julia check_Heun+SIB/heun_sib_combined_verification.jl --quick
```

Remove `--quick` for the reported production-size verification runs. `Plots.jl` is optional; numerical CSV and text output is still produced without it.

