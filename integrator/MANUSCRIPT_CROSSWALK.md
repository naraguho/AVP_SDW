# From the AVP manuscript to the Heun-SIB integrator

This document derives the numerical variables and update rules from the SDW sector of `AVP-dynamics3.pdf`. The order is deliberately the same as the physics:

1. construct the constrained electronic free energy;
2. obtain its conjugate magnetic field;
3. write the stochastic vector dynamics;
4. project that dynamics into amplitude and orientation;
5. discretize the amplitude with Heun and the orientation with SIB.

The manuscript equations referenced below are in Sec. VI and Appendix D.

## 1. Microscopic starting point

Section VI starts with the repulsive Hubbard model, manuscript Eq. (53):

```math
\hat H=-t\sum_{\langle ij\rangle,\alpha}
\left(\hat c_{i\alpha}^{\dagger}\hat c_{j\alpha}+\mathrm{H.c.}\right)
+U\sum_i\hat n_{i\uparrow}\hat n_{i\downarrow}.
```

The retained collective variables are the three components of the local electronic spin, manuscript Eq. (54):

```math
\hat s_i^a=\frac{1}{2}\sum_{\alpha\beta}
\hat c_{i\alpha}^{\dagger}\sigma^a_{\alpha\beta}\hat c_{i\beta},
\qquad a=x,y,z.
```

The prescribed SDW texture is the constrained expectation value

```math
\langle\hat{\mathbf s}_i\rangle=\mathbf m_i.
```

Thus $\mathbf m_i$ is not an auxiliary classical spin placed next to the electrons. It is the local spin density of the same constrained electronic state.

## 2. The field that drives the dynamics

For a prescribed texture $\{\mathbf m_i\}$, manuscript Eqs. (55)-(56) define a constrained finite-temperature Hartree-Fock problem. Once that problem is converged, its free energy is a function of the prescribed texture. The thermodynamic field is manuscript Eq. (57):

```math
\mathbf b_i=-\frac{\partial F}{\partial\mathbf m_i}.
```

This sign convention matters: $\mathbf b_i$ points in the direction that lowers the constrained free energy.

Appendix D distinguishes this thermodynamic field from the effective one-particle field $\mathbf h_i$ used during the Hartree-Fock iteration. Manuscript Eq. (D2) gives

```math
\mathbf h_i=\mathbf b_i-2U\mathbf m_i.
```

After convergence, manuscript Eq. (D7) converts the internal field back to the thermodynamic field:

```math
\mathbf b_i=\bar{\mathbf h}_i+2U\mathbf s_i[\bar G]
=\bar{\mathbf h}_i+2U\mathbf m_i.
```

Therefore `find_local_field!` must return $\mathbf b_i$, or its caller must perform this conversion. Passing $\mathbf h_i$ directly into the dynamical equation would omit $2U\mathbf m_i$.

The constrained iteration is manuscript Eqs. (D4)-(D6). The Fermi occupations in Eq. (D5) contain the chemical potential $\mu$, which is adjusted to maintain the chosen filling. Consequently, every electronic solve used by an old state, predictor state, or midpoint state must recalculate $\mu$.

## 3. Why the reversible term is a cross product

The local spin operators obey manuscript Eq. (58):

```math
[\hat s_i^a,\hat s_j^b]
=i\delta_{ij}\epsilon^{abc}\hat s_i^c.
```

This algebra generates the reversible AVP tensor and gives

```math
\left.\frac{d\mathbf m_i}{dt}\right|_{\mathrm{rev}}
=\mathbf m_i\times\mathbf b_i.
```

The cross product is therefore fixed by the microscopic spin commutator; it is not an independently chosen phenomenological precession term.

## 4. Longitudinal and transverse mobilities

The manuscript separates the moment into magnitude and direction in Eq. (59):

```math
\mathbf m_i=M_i\hat{\mathbf e}_i,
\qquad
M_i=|\mathbf m_i|,
\qquad
|\hat{\mathbf e}_i|=1.
```

For $M_i>0$, define the projectors

```math
P_{i,\parallel}=\hat{\mathbf e}_i\hat{\mathbf e}_i^{\mathsf T},
\qquad
P_{i,\perp}=I-P_{i,\parallel}.
```

Manuscript Eq. (60) is then

```math
\mathbf b_{i,\parallel}=P_{i,\parallel}\mathbf b_i
=\hat{\mathbf e}_i
(\hat{\mathbf e}_i\cdot\mathbf b_i),
\qquad
\mathbf b_{i,\perp}=P_{i,\perp}\mathbf b_i.
```

The dissipative mobility tensor is chosen locally as

```math
\Gamma_i=\Gamma_{\parallel}P_{i,\parallel}
+\Gamma_{\perp}P_{i,\perp},
```

so amplitude and orientation can relax on different timescales.

Combining reversible precession, dissipative response, and thermal forcing gives manuscript Eq. (61):

```math
\frac{d\mathbf m_i}{dt}
=\mathbf m_i\times\mathbf b_i
+\Gamma_{\parallel}\mathbf b_{i,\parallel}
+\Gamma_{\perp}\mathbf b_{i,\perp}
+\boldsymbol\eta_i(t).
```

This is the continuous vector equation from which the implemented split equations follow.

## 5. Deriving the amplitude equation

Because $\mathbf m_i=M_i\hat{\mathbf e}_i$, a Stratonovich differential obeys the ordinary chain rule:

```math
d\mathbf m_i=\hat{\mathbf e}_i,dM_i
+M_i,d\hat{\mathbf e}_i,
\qquad
\hat{\mathbf e}_i\cdot d\hat{\mathbf e}_i=0.
```

Dotting with $\hat{\mathbf e}_i$ gives

```math
dM_i=\hat{\mathbf e}_i\cdot d\mathbf m_i.
```

Now apply this projection term by term:

- $\hat{\mathbf e}_i\cdot(\mathbf m_i\times\mathbf b_i)=0$;
- $\hat{\mathbf e}_i\cdot\mathbf b_{i,\perp}=0$;
- $\hat{\mathbf e}_i\cdot\mathbf b_{i,\parallel}
  =\hat{\mathbf e}_i\cdot\mathbf b_i$.

The result is manuscript Eq. (62), repeated as Eq. (D11):

```math
\frac{dM_i}{dt}
=\Gamma_{\parallel}\hat{\mathbf e}_i\cdot\mathbf b_i
+\xi_i(t).
```

Appendix D, Eq. (D8), verifies that

```math
\frac{\partial F}{\partial M_i}
=-\hat{\mathbf e}_i\cdot\mathbf b_i.
```

Hence the deterministic amplitude drift is the Model-A relaxation
$-\Gamma_{\parallel}\partial F/\partial M_i$.

## 6. Deriving the orientation equation

Project the vector differential perpendicular to the moment. Manuscript Eq. (D10) gives

```math
d\hat{\mathbf e}_i
=\frac{1}{M_i}P_{i,\perp},d\mathbf m_i.
```

For the precession term,

```math
\frac{1}{M_i}P_{i,\perp}
(\mathbf m_i\times\mathbf b_i)
=\hat{\mathbf e}_i\times\mathbf b_i.
```

For transverse damping,

```math
\frac{\Gamma_{\perp}}{M_i}\mathbf b_{i,\perp}
=-\frac{\Gamma_{\perp}}{M_i}
\hat{\mathbf e}_i\times
(\hat{\mathbf e}_i\times\mathbf b_i),
```

where the vector identity
$\mathbf b_{i,\perp}=-\hat{\mathbf e}_i\times
(\hat{\mathbf e}_i\times\mathbf b_i)$ was used.

This yields manuscript Eq. (63), repeated as Eq. (D13):

```math
\frac{d\hat{\mathbf e}_i}{dt}
=\hat{\mathbf e}_i\times\mathbf b_i
-\frac{\Gamma_{\perp}}{M_i}
\hat{\mathbf e}_i\times
(\hat{\mathbf e}_i\times\mathbf b_i)
+\boldsymbol\zeta_i(t).
```

Both deterministic terms are tangent to the unit sphere. This is the geometric reason for applying SIB to $\hat{\mathbf e}_i$ instead of applying SIB to the full soft vector $\mathbf m_i$.

## 7. Noise and fluctuation-dissipation

Appendix D introduces independent longitudinal and transverse noise sources. The amplitude noise satisfies manuscript Eq. (D12):

```math
\langle\xi_i(t)\xi_j(t')\rangle
=2k_BT\Gamma_{\parallel}
\delta_{ij}\delta(t-t').
```

Its increment over a timestep $h$ is

```math
\sqrt{2k_BT\Gamma_{\parallel}}\,\Delta W_{i,\parallel},
\qquad
\Delta W_{i,\parallel}\sim N(0,h).
```

The transverse Stratonovich noise is

```math
\boldsymbol\zeta_i,dt
=\frac{\sqrt{2k_BT\Gamma_{\perp}}}{M_i}
P_{i,\perp}\circ d\mathbf W_i.
```

It has the covariance in manuscript Eq. (D14):

```math
\langle\zeta_i^a(t)\zeta_j^b(t')\rangle_{\mathbf m}
=\frac{2k_BT\Gamma_{\perp}}{M_i^2}
P_{i,\perp}^{ab}
\delta_{ij}\delta(t-t').
```

Since
$\boldsymbol\eta_i=\hat{\mathbf e}_i\xi_i+M_i\boldsymbol\zeta_i$,
the two independent noises reproduce manuscript Eq. (D15):

```math
\langle\eta_i^a(t)\eta_j^b(t')\rangle_{\mathbf m}
=2k_BT\left(
\Gamma_{\parallel}P_{i,\parallel}^{ab}
+\Gamma_{\perp}P_{i,\perp}^{ab}
\right)
\delta_{ij}\delta(t-t').
```

The factor $1/M_i$ in the orientation noise is essential. It converts transverse fluctuations of the moment into angular fluctuations while leaving $\Gamma_{\perp}$ as the transverse mobility of $\mathbf m_i$.

## 8. Heun discretization of the amplitude

Define the longitudinal drift evaluated from the complete texture:

```math
f_{M,i}(\mathbf m)
=\Gamma_{\parallel}
\hat{\mathbf e}_i\cdot\mathbf b_i[\{\mathbf m_j\}].
```

Let

```math
\sigma_{\parallel}=\sqrt{2k_BT\Gamma_{\parallel}}.
```

Using one stored Wiener increment, the Heun predictor is

```math
\widetilde M_i=M_{i,n}
+h f_{M,i}(\mathbf m_n)
+\sigma_{\parallel}\Delta W_{i,\parallel}.
```

After forming the predicted full texture and recomputing its constrained electronic field, the corrector is

```math
M_{i,n+1}=M_{i,n}
+\frac{h}{2}
\left[f_{M,i}(\mathbf m_n)+f_{M,i}(\widetilde{\mathbf m})\right]
+\sigma_{\parallel}\Delta W_{i,\parallel}.
```

The same $\Delta W_{i,\parallel}$ is used in both stages. Because the amplitude noise in Eq. (D11) is additive, it enters the final step once rather than being averaged twice.

## 9. Rewriting the orientation equation for SIB

SIB requires an equation in cross-product form. Define

```math
\mathbf a_i(\hat{\mathbf e}_i,M_i,\mathbf b_i)
=\mathbf b_i
-\frac{\Gamma_{\perp}}{M_i}
(\hat{\mathbf e}_i\times\mathbf b_i).
```

Then

```math
\hat{\mathbf e}_i\times\mathbf a_i
=\hat{\mathbf e}_i\times\mathbf b_i
-\frac{\Gamma_{\perp}}{M_i}
\hat{\mathbf e}_i\times
(\hat{\mathbf e}_i\times\mathbf b_i),
```

which is exactly the deterministic part of Eq. (D13).

For the noise, define its action on a Wiener increment by

```math
\boldsymbol\sigma_i\Delta\mathbf W_i
=-\frac{\sqrt{2k_BT\Gamma_{\perp}}}{M_i}
\hat{\mathbf e}_i\times\Delta\mathbf W_i.
```

Because

```math
\hat{\mathbf e}_i\times
[-\hat{\mathbf e}_i\times\Delta\mathbf W_i]
=P_{i,\perp}\Delta\mathbf W_i,
```

the resulting cross-product noise is exactly the tangential noise of Eq. (D14). Therefore Eq. (D13) becomes

```math
d\hat{\mathbf e}_i
=\hat{\mathbf e}_i\times\mathbf a_i,dt
+\hat{\mathbf e}_i\times
\boldsymbol\sigma_i\circ d\mathbf W_i.
```

## 10. SIB predictor and corrector

At the old state, form

```math
\mathbf q_i^n
=h\mathbf a_i^n
+\boldsymbol\sigma_i^n\Delta\mathbf W_i.
```

The SIB predictor is the implicit-midpoint rotation

```math
\widetilde{\mathbf e}_i-\mathbf e_{i,n}
=\frac{\mathbf e_{i,n}+\widetilde{\mathbf e}_i}{2}
\times\mathbf q_i^n.
```

Evaluate the drift and noise coefficients at the SIB predicted midpoint, reuse the same $\Delta\mathbf W_i$, and form $\mathbf q_i^p$. The corrector is

```math
\mathbf e_{i,n+1}-\mathbf e_{i,n}
=\frac{\mathbf e_{i,n}+\mathbf e_{i,n+1}}{2}
\times\mathbf q_i^p.
```

Each equation has the generic form

```math
\mathbf x^+-\mathbf x
=\frac{\mathbf x+\mathbf x^+}{2}\times\mathbf q.
```

Dotting it with $\mathbf x+\mathbf x^+$ proves
$|\mathbf x^+|=|\mathbf x|$. The `cayley` function in the verification code solves this equation analytically, so both the predictor and corrector preserve orientation length without post-step normalization.

## 11. Coupling the integrator to the electronic solver

For a state-dependent constrained field, a strict timestep has the following logical sequence:

1. Solve the constrained electronic problem at $\mathbf m_n$ and obtain $\mathbf b_n$ using Eq. (D7).
2. Draw one longitudinal and one transverse Wiener increment per site.
3. Construct the Heun amplitude predictor and SIB orientation predictor.
4. Construct the full predictor texture needed by the Heun amplitude corrector and recompute its thermodynamic field.
5. Construct the predicted midpoint required by SIB and recompute the thermodynamic field used by the SIB corrector.
6. Complete both correctors with the same Wiener increments drawn in step 2.
7. Recombine $\mathbf m_{i,n+1}=M_{i,n+1}\hat{\mathbf e}_{i,n+1}$.

At every constrained electronic solve, the spin constraints must converge, $\mu$ must be readjusted to maintain filling, and the converged internal field must be converted to $\mathbf b_i$ with Eq. (D7).

The manuscript specifies Heun, SIB, and noise reuse, but it does not explicitly enumerate these distinct electronic-field evaluation configurations. For a constant field they coincide. For the general SDW field they need not coincide. Reusing only one endpoint-predictor field for both correctors is therefore an additional approximation and should be stated and tested by timestep refinement.

## 12. What the verification folders test

- [`check_SIB/`](check_SIB/) isolates Eq. (D13) and the SIB algorithm. It tests unit-length preservation, deterministic order two, stochastic strong order one-half, stochastic weak order one, phase error, and fixed-length conservation properties.
- [`check_Heun+SIB/`](check_Heun+SIB/) combines a Heun soft amplitude with a SIB orientation. It verifies the limiting convergence orders of the reconstructed vector $\mathbf m=M\mathbf e$.

The manufactured combined test does not run the constrained electronic calculation, so it does not verify Eqs. (D2)-(D7). The production SDW code still needs residual checks, filling checks, and a coupled-noise timestep-refinement study.

## References used in Appendix D

- J. L. Garcia-Palacios and F. J. Lazaro, “Langevin-dynamics study of the dynamical properties of small magnetic particles,” *Physical Review B* **58**, 14937 (1998). This is manuscript Ref. [125] for the Stratonovich stochastic Landau-Lifshitz formulation.
- J. H. Mentink, M. V. Tretyakov, A. Fasolino, M. I. Katsnelson, and Th. Rasing, “Stable and fast semi-implicit integration of the stochastic Landau-Lifshitz equation,” *Journal of Physics: Condensed Matter* **22**, 176001 (2010), [doi:10.1088/0953-8984/22/17/176001](https://doi.org/10.1088/0953-8984/22/17/176001). This is manuscript Ref. [126] and the source of SIB.
