# Longitudinal-Heun / transverse-SIB SDW integrator

This methodology implements the SDW dynamics of Sec. VI, Eqs. (57)-(63), and Appendix D, Eqs. (D2)-(D15), of `AVP-dynamics3.pdf`. The detailed [manuscript crosswalk](MANUSCRIPT_CROSSWALK.md) traces the constrained electronic field, fluctuation-dissipation relations, Heun amplitude step, and SIB orientation step equation by equation.

## 1. State and field convention

At each lattice site, write the soft SDW vector as

$$
\mathbf m=M\mathbf e,\qquad M=|\mathbf m|,\qquad |\mathbf e|=1.
$$

Let $\mathbf b=-\partial F/\partial\mathbf m$ be the thermodynamic driving field returned by the constrained electronic calculation. This sign convention must be used consistently in both the equations and code. Define

$$
b_\parallel=\mathbf e\cdot\mathbf b,\qquad
\mathbf b_\parallel=b_\parallel\mathbf e,\qquad
\mathbf b_\perp=(I-\mathbf e\mathbf e^{\mathsf T})\mathbf b.
$$

The stochastic generalized Landau-Lifshitz dynamics used here is

$$
d\mathbf m=
\left[\mathbf m\times\mathbf b+
\Gamma_\parallel\mathbf b_\parallel+
\Gamma_\perp\mathbf b_\perp\right]dt
+\sqrt{2k_BT\Gamma_\parallel}\,\mathbf e\,dW_\parallel
+\sqrt{2k_BT\Gamma_\perp}\,P_\perp\circ d\mathbf W_\perp,
$$

where $P_\perp=I-\mathbf e\mathbf e^{\mathsf T}$. The transverse stochastic integral is interpreted in the Stratonovich sense. Independent longitudinal and transverse noises give the required projected covariance.

Projecting parallel and perpendicular to $\mathbf e$ gives

$$
dM=\Gamma_\parallel b_\parallel\,dt+
\sqrt{2k_BT\Gamma_\parallel}\,dW_\parallel,
$$

and, away from $M=0$,

$$
d\mathbf e=\left[\mathbf e\times\mathbf b+
\frac{\Gamma_\perp}{M}\mathbf b_\perp\right]dt+
\frac{\sqrt{2k_BT\Gamma_\perp}}{M}P_\perp\circ d\mathbf W_\perp.
$$

Every term in the orientation equation is tangent to the unit sphere. This is why a geometric spin method is appropriate for $\mathbf e$, whereas the scalar amplitude should not be constrained by SIB.

## 2. Why Heun for amplitude and SIB for direction

The amplitude is a scalar soft mode: it must be allowed to relax and fluctuate. A stochastic Heun step evaluates its drift at an old-state and predictor-state field and reuses the same scalar Wiener increment in both stages.

The direction is a rotational mode. An explicit Heun predictor does not preserve unit length at its predictor stage. SIB instead retains the implicit-midpoint rotational form in both predictor and corrector. For an equation written as

$$
d\mathbf e=\mathbf e\times\mathbf a(\mathbf e)dt+
\mathbf e\times\sigma(\mathbf e)\circ d\mathbf W,
$$

SIB uses

$$
\widetilde{\mathbf e}=\mathbf e_n+
\frac{\mathbf e_n+\widetilde{\mathbf e}}{2}\times
\left[h\mathbf a(\mathbf e_n)+\sigma(\mathbf e_n)\Delta\mathbf W\right],
$$

followed by

$$
\mathbf e_{n+1}=\mathbf e_n+
\frac{\mathbf e_n+\mathbf e_{n+1}}{2}\times
\left[h\mathbf a(\mathbf e_{n+1/2}^{p})+
\sigma(\mathbf e_{n+1/2}^{p})\Delta\mathbf W\right],
\quad
\mathbf e_{n+1/2}^{p}=\frac{\mathbf e_n+\widetilde{\mathbf e}}{2}.
$$

The same Wiener increment is used in both stages. Each implicit vector equation is a three-dimensional rotation and can be solved analytically with the Cayley transform; no iterative nonlinear solve is required for the spin rotation itself.

## 3. One production timestep

For every site at time $t_n$:

1. Set $M_n=|\mathbf m_n|$ and $\mathbf e_n=\mathbf m_n/M_n$. Handle the small-$M$ case explicitly as described below.
2. Solve the constrained electronic problem at $\mathbf m_n$, recalculating the chemical potential after every diagonalization so the target filling is maintained. Obtain $\mathbf b_n$.
3. Draw one scalar $\Delta W_\parallel\sim N(0,h)$ and one three-vector $\Delta\mathbf W_\perp\sim N(\mathbf0,hI)$ per site. Store them; do not redraw them in the corrector.
4. Form the longitudinal Heun predictor $\widetilde M$.
5. Form the transverse SIB predictor $\widetilde{\mathbf e}$ with an implicit-midpoint/Cayley rotation.
6. Build the predicted midpoint configuration required by SIB. Recompute the constrained electronic field and chemical potential at that evaluation configuration. For a state-dependent field, evaluating only at the end predictor is not algebraically identical to the SIB midpoint in Mentink et al.
7. Correct $M$ with the Heun trapezoidal drift average, using the stored $\Delta W_\parallel$.
8. Correct $\mathbf e$ with the second SIB midpoint rotation, using the stored $\Delta\mathbf W_\perp$.
9. Recombine $\mathbf m_{n+1}=M_{n+1}\mathbf e_{n+1}$, and record diagnostics before applying any physical amplitude bound.

The electronic field tolerances must be appreciably smaller than the change produced by halving $h$; otherwise solver error masks the integrator order.

## 4. Numerical edge cases

- **Small amplitude.** Direction is undefined at $M=0$, while the orientation equation contains $1/M$. Choose and document a threshold. A defensible implementation either takes a full-vector step in this region or decreases the timestep; silently replacing $M$ by a relatively large floor changes the stochastic model.
- **Amplitude sign.** If $M$ is defined as a magnitude, a scalar step that crosses zero requires a convention (for example, flip $\mathbf e$ and use $|M|$). This must be tested rather than hidden by clipping.
- **Electronic bound.** The microscopic definition may impose $|\mathbf m|\leq 1/2$. Hard clipping changes transition probabilities. Count clipping events and demonstrate that they disappear under timestep refinement, or formulate boundary-consistent dynamics.
- **Noise reuse.** Predictor and corrector must use identical Wiener increments. Drawing fresh noise changes the method.
- **Filling constraint.** Recalculate $\mu$ at every old, predictor-midpoint, and other field evaluation—not once at initialization.

## 5. Acceptance protocol for the full SDW code

The isolated tests in [`check_SIB/`](check_SIB/) and [`check_Heun+SIB/`](check_Heun+SIB/) verify the mathematical integrators. Before production, run the actual SDW calculation with $h,h/2,h/4$ using coupled Brownian paths. Tighten the electronic solver until further tightening no longer changes the timestep differences. Compare the physical vector $\mathbf m$, energy, filling, order parameters, and the rate of small-amplitude or clipping events. A timestep is acceptable when discretization changes are below the statistical uncertainty of the observables being reported.

## Reference

J. H. Mentink, M. V. Tretyakov, A. Fasolino, M. I. Katsnelson, and Th. Rasing, “Stable and fast semi-implicit integration of the stochastic Landau–Lifshitz equation,” *Journal of Physics: Condensed Matter* **22**, 176001 (2010). [doi:10.1088/0953-8984/22/17/176001](https://doi.org/10.1088/0953-8984/22/17/176001), [arXiv:1002.1801](https://arxiv.org/abs/1002.1801).

The predictor/corrector equations above are the SIB structure of Eq. (18) in that work, translated into the present direction variable.
