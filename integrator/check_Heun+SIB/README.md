# Strict three-field Heun + SIB: J1-J2-a demonstration

This folder demonstrates the strict three-field integrator for a vector `m = M e` whose spin amplitude `M` can vary. Heun advances the amplitude and SIB advances the unit direction.

In the manuscript, this corresponds to composing the amplitude equation Eq. (62)/Eq. (D11), integrated by Heun, with the orientation equation Eq. (63)/Eq. (D13), integrated by SIB as specified in Appendix D.3.

## 1. Reference algorithm: three field evaluations per timestep

The reference implementation treats amplitude and orientation as two components of one coupled state. Their predictors advance over the same interval, but their correctors require fields at different configurations. For a state-dependent electronic field, one timestep therefore contains **three constrained field evaluations**.

Start from

```math
\mathbf m_n=M_n\mathbf e_n,
\qquad |\mathbf e_n|=1.
```

### 1. Old-state field

Solve the constrained electronic problem at the old texture:

```math
\mathbf b_n=\mathbf b[M_n\mathbf e_n].
```

The chemical potential is readjusted during this and every subsequent electronic solve so that the requested filling is maintained.

### 2. Draw noise once

Draw independent longitudinal and transverse Wiener increments:

```math
\Delta W_\parallel\sim N(0,h),
\qquad
\Delta\mathbf W_\perp\sim N(\mathbf0,hI).
```

The same increments are reused in the predictor and corrector stages. Redrawing noise in a corrector would define a different stochastic method.

### 3. Joint predictor

Define the longitudinal drift

```math
f_M(M,\mathbf e,\mathbf b)
=\Gamma_\parallel\mathbf e\cdot\mathbf b.
```

The Heun amplitude predictor is

```math
\widetilde M
=M_n+h f_M(M_n,\mathbf e_n,\mathbf b_n)
+\sqrt{2k_BT\Gamma_\parallel}\,\Delta W_\parallel.
```

For the orientation, form

```math
\mathbf a_n
=\mathbf b_n
-\frac{\Gamma_\perp}{M_n}
(\mathbf e_n\times\mathbf b_n),
```

include the transverse stochastic rotation in `q_n`, and solve the first SIB equation

```math
\widetilde{\mathbf e}-\mathbf e_n
=\frac{\mathbf e_n+\widetilde{\mathbf e}}{2}
\times\mathbf q_n
```

with the Cayley formula. Consequently,

```math
|\widetilde{\mathbf e}|=1
```

without normalization. The pair `(M_tilde,e_tilde)` is a joint predictor for time `t_(n+1)`.

### 4. Heun corrector at the predicted endpoint

Construct the predicted endpoint texture and perform the second field solve:

```math
\widetilde{\mathbf m}=\widetilde M\widetilde{\mathbf e},
\qquad
\mathbf b_H=\mathbf b[\widetilde{\mathbf m}].
```

Then

```math
M_{n+1}=M_n
+\frac h2\left[
\Gamma_\parallel\mathbf e_n\cdot\mathbf b_n
+\Gamma_\parallel\widetilde{\mathbf e}\cdot\mathbf b_H
\right]
+\sqrt{2k_BT\Gamma_\parallel}\,\Delta W_\parallel.
```

This is ordinary Heun applied to a drift that depends on the complete coupled state, not only on the scalar amplitude.

### 5. SIB corrector at the predicted midpoint

Construct predictor-based midpoint variables:

```math
M_S=\frac{M_n+\widetilde M}{2},
\qquad
\mathbf e_S=\frac{\mathbf e_n+\widetilde{\mathbf e}}{2},
\qquad
\mathbf m_S=M_S\mathbf e_S.
```

The midpoint direction is an evaluation point and is not normalized. Perform the third field solve:

```math
\mathbf b_S=\mathbf b[\mathbf m_S].
```

Evaluate the SIB drift and noise coefficients at `(M_S,e_S,b_S)` and solve

```math
\mathbf e_{n+1}-\mathbf e_n
=\frac{\mathbf e_n+\mathbf e_{n+1}}{2}
\times\mathbf q_S.
```

The Cayley solution gives unit final orientation algebraically. Finally recombine once:

```math
\mathbf m_{n+1}=M_{n+1}\mathbf e_{n+1}.
```

Thus the required fields are

```math
\boxed{\mathbf b_n,\quad\mathbf b_H,\quad\mathbf b_S}.
```

Using `b_H` in the SIB corrector is a cheaper approximation, not the literal midpoint evaluation required by SIB.

## 2. Heun for the spin amplitude

Begin with a scalar Stratonovich stochastic differential equation:

```math
dM=f(M,t)\,dt+g(M,t)\circ dW.
```

For a step of length `h`, draw one Wiener increment

```math
\Delta W_n\sim N(0,h).
```

Heun first forms an Euler predictor:

```math
\widetilde M
=M_n+h f(M_n,t_n)+g(M_n,t_n)\Delta W_n.
```

It then averages the drift and diffusion coefficients at the old and predicted endpoints:

```math
M_{n+1}=M_n
+\frac{h}{2}\left[f(M_n,t_n)+f(\widetilde M,t_{n+1})\right]
+\frac{\Delta W_n}{2}
\left[g(M_n,t_n)+g(\widetilde M,t_{n+1})\right].
```

The same random increment appears in both stages. This predictor-corrector is the stochastic analogue of the explicit trapezoidal rule and is consistent with the Stratonovich interpretation.

## 3. Combining Heun with SIB

One timestep advances two different geometries:

1. Heun advances the spin amplitude.
2. SIB advances the unit orientation through two length-preserving midpoint rotations.
3. The two updated variables are recombined only at the end:

```math
\mathbf m_{i,n+1}=M_{i,n+1}\hat{\mathbf e}_{i,n+1}.
```

The full vector must not be normalized after recombination: doing so would erase the amplitude dynamics. Likewise, SIB should not be applied to the amplitude, because its purpose is precisely to preserve the direction length.

At finite temperature the two methods do not have the same strong convergence order. For this problem, additive-noise Heun gives approximately strong order one, while the SIB orientation with noncommuting multiplicative noise gives strong order one half. The error of the reconstructed vector is therefore expected to be limited by the SIB sector.

## 4. Physical benchmark: variable-amplitude J1-J2-a model

The primary physics demonstration is [`j1j2a_strict_physics.jl`](j1j2a_strict_physics.jl). It uses the strict three-field algorithm on

```math
F=J_1\sum_{\langle ij\rangle}\mathbf m_i\cdot\mathbf m_j
+J_2\sum_{\langle\!\langle ij\rangle\!\rangle}\mathbf m_i\cdot\mathbf m_j
+\frac a4\sum_i(|\mathbf m_i|^2-b)^2.
```

Unlike a fixed-length model, this system tests both pieces of the integrator: SIB rotates each unit direction while Heun changes the local amplitude.

For a uniform-amplitude Néel state,

```math
M_N^2=b+\frac{4(J_1-J_2)}a,
\qquad
\frac{F_N}{N}=(-2J_1+2J_2)M_N^2
+\frac a4(M_N^2-b)^2.
```

For a stripe state,

```math
M_S^2=b+\frac{4J_2}a,
\qquad
\frac{F_S}{N}=-2J_2M_S^2
+\frac a4(M_S^2-b)^2.
```

The two analytic branches cross at `J2/J1=1/2`. This supplies a physically interpretable benchmark rather than only a numerical convergence slope.

![J1-J2-a phase and spin-amplitude benchmark](results/j1j2a_strict_physics.png)

The numerical relaxation reproduces three linked pieces of physics:

- the Néel-to-stripe crossing at `J2/J1=1/2`;
- the analytic relaxed amplitudes on both sides of the transition;
- the change from `Q(pi,pi)` order to stripe `Q(pi,0)` or `Q(0,pi)` order.

For clarity, the simulation uses only the corresponding ordered state in each regime: Néel initialization for `J2/J1 < 1/2` and stripe initialization for `J2/J1 >= 1/2`. At the degenerate point `J2/J1 = 1/2`, the run starts near the stripe state. Both analytic branches are shown over the full range so that their crossing remains visible.

Run the full benchmark with

```bash
julia j1j2a_strict_physics.jl
```

or use `--quick` for a short smoke test. The additional scripts in this directory remain available as implementation-level checks, while this README focuses on the physical J1-J2-a demonstration.
