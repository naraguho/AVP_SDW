# Connection to the SDW sector of the AVP manuscript

This note connects the numerical scheme in this repository directly to Sec. VI and Appendix D of `AVP-dynamics3.pdf`. It separates three layers that should not be conflated:

1. the constrained electronic calculation that returns the thermodynamic field;
2. the continuous stochastic SDW equations;
3. the discrete longitudinal-Heun/transverse-SIB integrator.

## 1. Section VI: microscopic field and SDW dynamics

The manuscript begins from the Hubbard Hamiltonian [Eq. (53)] and local electronic spin operators [Eq. (54)]. The prescribed collective variable is the constrained spin expectation

$$
\langle\hat{\mathbf s}_i\rangle=\mathbf m_i.
$$

The constrained Hartree-Fock functional and quadratic Hamiltonian are Eqs. (55) and (56). After the electronic constraint is converged, the field used by the dynamics is

$$
\mathbf b_i=-\frac{\partial F}{\partial\mathbf m_i},
\tag{57}
$$

not merely the effective field appearing inside an unconverged one-particle Hamiltonian.

The local spin commutator [Eq. (58)] generates the reversible contribution $\mathbf m_i\times\mathbf b_i$. The amplitude-orientation variables and field projections are

$$
\mathbf m_i=M_i\hat{\mathbf e}_i,
\qquad M_i=|\mathbf m_i|,
\qquad |\hat{\mathbf e}_i|=1,
\tag{59}
$$

$$
\mathbf b_{i,\parallel}=\hat{\mathbf e}_i
(\hat{\mathbf e}_i\cdot\mathbf b_i),
\qquad
\mathbf b_{i,\perp}=\mathbf b_i-\mathbf b_{i,\parallel}.
\tag{60}
$$

The full vector equation is Eq. (61):

$$
\frac{d\mathbf m_i}{dt}
=\mathbf m_i\times\mathbf b_i
+\Gamma_\parallel\mathbf b_{i,\parallel}
+\Gamma_\perp\mathbf b_{i,\perp}
+\boldsymbol\eta_i(t).
\tag{61}
$$

Its longitudinal and transverse projections give the two equations integrated by this repository:

$$
\frac{dM_i}{dt}
=\Gamma_\parallel\hat{\mathbf e}_i\cdot\mathbf b_i+\xi_i(t),
\tag{62}
$$

$$
\frac{d\hat{\mathbf e}_i}{dt}
=\hat{\mathbf e}_i\times\mathbf b_i
-\frac{\Gamma_\perp}{M_i}
\hat{\mathbf e}_i\times
(\hat{\mathbf e}_i\times\mathbf b_i)
+\boldsymbol\zeta_i(t).
\tag{63}
$$

Thus Heun is applied only to the soft amplitude $M_i$, while SIB is applied only to the unit orientation $\hat{\mathbf e}_i$.

## 2. Appendix D: what the electronic solver must return

Appendix D distinguishes the effective field $\mathbf h_i$ used inside the Hartree-Fock Hamiltonian from the thermodynamic field $\mathbf b_i$ used in Eqs. (61)-(63):

$$
\mathbf h_i=\mathbf b_i-2U\mathbf m_i,
\tag{D2}
$$

$$
\mathbf b_i=\bar{\mathbf h}_i+2U\mathbf s_i[\bar G]
=\bar{\mathbf h}_i+2U\mathbf m_i.
\tag{D7}
$$

Accordingly, a production routine such as `find_local_field!` must return $\mathbf b_i$, or its caller must perform the conversion in Eq. (D7). Supplying $\mathbf h_i$ directly to the dynamical equation would omit the $2U\mathbf m_i$ contribution.

Equations (D4)-(D6) define the constrained iteration. In Eq. (D5), the chemical potential is adjusted to maintain the requested filling. This adjustment is required at every electronic diagonalization performed for an old state, predictor state, or SIB midpoint state.

Equations (D8)-(D10) then show that the amplitude force and tangential force are projections of the same $\mathbf b_i$:

$$
\frac{\partial F}{\partial M_i}
=-\hat{\mathbf e}_i\cdot\mathbf b_i,
\qquad
\nabla_{S^2,i}F=-M_i\mathbf b_{i,\perp}.
\tag{D8-D9}
$$

This is why the longitudinal and transverse updates must use fields obtained from the same constrained free-energy problem.

## 3. Appendix D: fluctuation-dissipation structure

The amplitude noise in Eqs. (D11)-(D12) satisfies

$$
\langle\xi_i(t)\xi_j(t')\rangle
=2k_BT\Gamma_\parallel\delta_{ij}\delta(t-t').
\tag{D12}
$$

For a timestep $h$, its discrete increment is therefore

$$
\sqrt{2k_BT\Gamma_\parallel}\,\Delta W_{i,\parallel},
\qquad \Delta W_{i,\parallel}\sim N(0,h).
$$

The orientational equation is repeated as Eq. (D13). Its explicit Stratonovich noise representation is

$$
\boldsymbol\zeta_i,dt
=\frac{\sqrt{2k_BT\Gamma_\perp}}{M_i}
P_{i,\perp}\circ d\mathbf W_i,
\qquad
P_{i,\perp}=I-\hat{\mathbf e}_i\hat{\mathbf e}_i^{\mathsf T}.
$$

This produces Eq. (D14),

$$
\langle\zeta_i^a(t)\zeta_j^b(t')\rangle_{\mathbf m}
=\frac{2k_BT\Gamma_\perp}{M_i^2}
P_{i,\perp}^{ab}\delta_{ij}\delta(t-t'),
\tag{D14}
$$

and, with $\boldsymbol\eta_i=\hat{\mathbf e}_i\xi_i+M_i\boldsymbol\zeta_i$, the vector covariance in Eq. (D15):

$$
\langle\eta_i^a(t)\eta_j^b(t')\rangle_{\mathbf m}
=2k_BT\left(
\Gamma_\parallel P_{i,\parallel}^{ab}
+\Gamma_\perp P_{i,\perp}^{ab}
\right)\delta_{ij}\delta(t-t').
\tag{D15}
$$

The factors $1/M_i$ in the orientation drift and noise are therefore required. They ensure that $\Gamma_\perp$ remains the transverse mobility of the magnetic moment rather than an unrelated angular mobility.

## 4. Discrete amplitude step: Heun for Eq. (D11)

Define

$$
f_{M,i}(\mathbf M,\hat{\mathbf e})
=\Gamma_\parallel\hat{\mathbf e}_i\cdot\mathbf b_i[\{M_j\hat{\mathbf e}_j\}],
\qquad
\sigma_\parallel=\sqrt{2k_BT\Gamma_\parallel}.
$$

With one stored Wiener increment, the stochastic Heun step is

$$
\widetilde M_i=M_{i,n}+h f_{M,i}^{n}
+\sigma_\parallel\Delta W_{i,\parallel},
$$

$$
M_{i,n+1}=M_{i,n}
+\frac{h}{2}\left(f_{M,i}^{n}+f_{M,i}^{p}\right)
+\sigma_\parallel\Delta W_{i,\parallel}.
$$

Here $f^p$ uses the thermodynamic field recomputed at the predicted full magnetic configuration. The noise is additive in Eq. (D11), so it appears once in the final update; the same realization is used to construct the predictor.

## 5. Discrete orientation step: SIB for Eq. (D13)

Equation (D13) can be written in the cross-product form required by SIB:

$$
d\hat{\mathbf e}_i
=\hat{\mathbf e}_i\times\mathbf a_i\,dt
+\hat{\mathbf e}_i\times
\boldsymbol\sigma_i\circ d\mathbf W_i,
$$

with

$$
\mathbf a_i
=\mathbf b_i-rac{\Gamma_\perp}{M_i}
(\hat{\mathbf e}_i\times\mathbf b_i),
$$

$$
\boldsymbol\sigma_i\Delta\mathbf W_i
=-\frac{\sqrt{2k_BT\Gamma_\perp}}{M_i}
\hat{\mathbf e}_i\times\Delta\mathbf W_i.
$$

Indeed, $\hat{\mathbf e}\times[-\hat{\mathbf e}\times\Delta\mathbf W]
=P_\perp\Delta\mathbf W$, reproducing Eq. (D14).

For the SIB predictor, define

$$
\mathbf q_i^n=h\mathbf a_i^n
+\boldsymbol\sigma_i^n\Delta\mathbf W_i,
$$

and solve

$$
\widetilde{\mathbf e}_i-\mathbf e_{i,n}
=\frac{\mathbf e_{i,n}+\widetilde{\mathbf e}_i}{2}
\times\mathbf q_i^n.
$$

For the corrector, evaluate the drift and diffusion coefficients at the SIB predicted midpoint, reuse the same $\Delta\mathbf W_i$, and solve

$$
\mathbf e_{i,n+1}-\mathbf e_{i,n}
=\frac{\mathbf e_{i,n}+\mathbf e_{i,n+1}}{2}
\times\mathbf q_i^p.
$$

Both implicit equations are solved analytically by the Cayley map used in the verification code. Consequently, each SIB stage preserves $|\hat{\mathbf e}_i|$ to floating-point roundoff without an external normalization step.

## 6. Coupling to a state-dependent SDW field

Appendix D states that the amplitude uses Heun, the orientation uses SIB, and both stages reuse the same noise realization. It does not spell out every electronic-field evaluation required when $\mathbf b_i$ depends nonlocally on the complete predicted texture.

A literal implementation of both named methods requires distinguishing two evaluation configurations:

- the Heun amplitude corrector needs the field at the predicted endpoint configuration;
- the SIB orientation corrector needs its coefficients at the SIB predicted midpoint configuration.

These configurations coincide for the constant-field manufactured test but not for a general constrained SDW field. Reusing a single endpoint field for both correctors is an additional approximation, not the exact SIB staging of Mentink et al. If this shortcut is used for cost reasons, it must be identified and tested by timestep refinement. A strict implementation may require separate constrained electronic solves for the Heun predictor endpoint and SIB midpoint.

At every such solve:

1. enforce the spin constraints of Eq. (D6);
2. readjust $\mu$ in Eq. (D5) to maintain filling;
3. convert the converged $\bar{\mathbf h}_i$ to $\mathbf b_i$ with Eq. (D7);
4. keep the same Wiener increments throughout the full predictor-corrector step.

## 7. What each verification folder establishes

| Folder | Manuscript connection | Claim tested |
|---|---|---|
| [`check_SIB/`](check_SIB/) | Eq. (D13) and the SIB prescription in Appendix D.3 | unit-length preservation, deterministic order two, stochastic strong order one-half, stochastic weak order one, and fixed-length conservation tests |
| [`check_Heun+SIB/`](check_Heun+SIB/) | composition of Eqs. (D11) and (D13) | limiting orders of the reconstructed vector $\mathbf m=M\mathbf e$ and separation of longitudinal and transverse updates |

The combined manufactured test does not validate Eqs. (D2)-(D7), because it does not run the constrained electronic solver. That part must be checked in the production SDW code through residual tests, filling conservation, and coupled timestep refinement.

## References used by the manuscript

- J. L. García-Palacios and F. J. Lázaro, “Langevin-dynamics study of the dynamical properties of small magnetic particles,” *Physical Review B* **58**, 14937 (1998). This is Ref. [125] for the Stratonovich stochastic Landau-Lifshitz formulation.
- J. H. Mentink, M. V. Tretyakov, A. Fasolino, M. I. Katsnelson, and Th. Rasing, “Stable and fast semi-implicit integration of the stochastic Landau-Lifshitz equation,” *Journal of Physics: Condensed Matter* **22**, 176001 (2010), [doi:10.1088/0953-8984/22/17/176001](https://doi.org/10.1088/0953-8984/22/17/176001). This is Ref. [126] and the source of SIB.

