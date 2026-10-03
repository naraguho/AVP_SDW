# Reflected-amplitude stochastic SDW dynamics

This directory records the current finite-temperature SDW experiment on a
$16\times16$ square lattice. It combines four ingredients:

1. direct stochastic evolution of the local amplitude
   $M_i=|\mathbf m_i|$ on its physical interval $0\le M_i\le1/2$;
2. a reflected stochastic-Heun predictor and corrector for $M_i$;
3. the original two-stage SIB/Cayley update for the direction
   $\mathbf e_i=\mathbf m_i/M_i$;
4. a local, site-resolved adaptive mixing coefficient $\alpha_i$ for the
   constrained electronic-field inversion.

The two examples below use the same temperature in the electronic
Fermi-Dirac distribution and in the stochastic bath. The $T=0.01$ snapshot
shows a clear checkerboard Néel pattern, primarily in $m_x$. The $T=0.03$
snapshot has no comparable system-wide alternating pattern and has a much
broader, lower-amplitude spin-length texture. These are representative
finite-size, finite-time configurations. Establishing a thermodynamic phase
boundary requires ensemble averages, equilibration tests, and finite-size
scaling.

## Representative final configurations

### Matched temperatures $T_{\rm elec}=T_{\rm fluc}=0.01$

![L=16 reflected-SDE final configuration at T=0.01](results/Neel16_T0.01_components_final.png)

At step 7620 ($t=381$), the displayed configuration has
$\overline M=0.30782$ and $M_{\max}=0.41019$. The alternating sign of $m_x$
across neighboring sites is direct visual evidence of finite-size Néel order
in this trajectory.

### Matched temperatures $T_{\rm elec}=T_{\rm fluc}=0.03$

![L=16 reflected-SDE final configuration at T=0.03](results/Neel16_T0.03_components_final.png)

At step 10000 ($t=500$), the displayed configuration has
$\overline M=0.20117$ and $M_{\max}=0.44044$. None of the three Cartesian
components shows the coherent lattice-wide checkerboard pattern seen at
$T=0.01$.

The comparison is important because the stochastic bath is not being turned
off while the electronic temperature is varied. Both calculations use

$$
T_{\rm elec}=T_{\rm fluc},
$$

so the electronic occupation and order-parameter noise are evaluated at the
same nominal temperature.

## Reflected flat-$M$ model

We decompose each local SDW vector as

$$
\mathbf m_i=M_i\mathbf e_i,
\qquad 0\le M_i\le\frac12,
\qquad |\mathbf e_i|=1.
$$

The amplitude is treated as a scalar collective coordinate with a flat-$M$
measure. Its continuous reflected Langevin model is

$$
dM_i=
\Gamma_\parallel\lambda_{\parallel,i}\,dt
+\sqrt{2T_{\rm fluc}\Gamma_\parallel}\,dW_{i,\parallel}
+dK_i^{(0)}-dK_i^{(1/2)},
$$

where

$$
\lambda_{\parallel,i}=\mathbf e_i\cdot\boldsymbol\lambda_i,
\qquad
\boldsymbol\lambda_i=2\mathbf m_i-\mathbf B_i.
$$

$K_i^{(0)}$ and $K_i^{(1/2)}$ are boundary terms that act only at the lower
and upper walls. Because this is a flat-$M$ model rather than a radial rewrite
of additive Cartesian noise, no $2T\Gamma_\parallel/M_i$ geometric drift is
included.

The finite-step reflection map is

$$
R(y)=U-\left|\operatorname{mod}(y,2U)-U\right|,
\qquad U=\frac12.
$$

For example, an unconstrained proposal $M^{\rm raw}=0.51$ is mapped to

$$
R(0.51)=1-0.51=0.49.
$$

This is a mirror reflection, not clipping or rejection: the Gaussian draw is
retained and the overshoot is folded back into the physical interval.

## Reflected stochastic Heun plus SIB

With

$$
\eta_i=\sqrt{2T_{\rm fluc}\Gamma_\parallel\Delta t}\,\xi_i,
\qquad \xi_i\sim\mathcal N(0,1),
$$

the amplitude predictor is

$$
M_i^p=R\!\left[
M_i^n+\Gamma_\parallel\lambda_{\parallel,i}^n\Delta t+\eta_i
\right].
$$

The constrained electronic field is recomputed at the physical predictor
texture $\mathbf m_i^p=M_i^p\mathbf e_i^p$. The corrected amplitude is

$$
M_i^{n+1}=R\!\left[
M_i^n+
\frac{\Gamma_\parallel\Delta t}{2}
\left(\lambda_{\parallel,i}^n+\lambda_{\parallel,i}^p\right)
+\eta_i
\right].
$$

The same longitudinal Wiener increment $\eta_i$ is reused in the predictor
and corrector. The predictor is a trial state, not an additional physical
step. Reflection is applied there so that the expensive electronic inversion
is never asked to match an unphysical target with $M_i>1/2$. The final
corrector is reflected independently so that the accepted state also remains
physical.

The orientation $\mathbf e_i$ is evolved with the two-stage SIB method. Its
implicit midpoint rotations are evaluated with a Cayley map, preserving
$|\mathbf e_i|=1$ without post-step normalization. One dynamical step uses
separate constrained-field solves for the Heun endpoint, SIB midpoint, and
accepted state.

The mirror construction is motivated by the symmetrized Euler method of
M. Bossy, E. Gobet, and D. Talay, *A Symmetrized Euler Scheme for an Efficient
Approximation of Reflected Diffusions*, Journal of Applied Probability 41,
877–889 (2004), [doi:10.1239/jap/1091543431](https://doi.org/10.1239/jap/1091543431).
That paper proves a weak-convergence result for its Euler construction. Our
stagewise reflected Heun/SIB composition is an extension and therefore still
requires direct timestep-refinement tests in the coupled SDW calculation.

## Site-resolved adaptive $\alpha_i$

The constrained field solves

$$
\langle\mathbf m_i\rangle_{\mathbf B}=\mathbf m_i^{\rm target}
$$

by updating every site with its own mixing coefficient $\alpha_i$. Define the
local field and response changes between two inversion iterates as

$$
\mathbf s_i=\Delta\mathbf B_i,
\qquad
\mathbf y_i=\Delta\langle\mathbf m_i\rangle.
$$

When $\mathbf s_i\cdot\mathbf y_i>0$ and their directional alignment is at
least 0.5, the code forms the inverse-response estimate

$$
\alpha_i^{\rm sec}=
\frac{\mathbf s_i\cdot\mathbf s_i}
     {\mathbf s_i\cdot\mathbf y_i}.
$$

Every five field iterations, this candidate is clipped to $[0.5,64]$ and
geometrically blended with the current $\alpha_i$. A single accepted increase
is limited to a factor 1.5. If the local residual worsens by more than 2% for
two consecutive iterations, $\alpha_i$ is reduced by a factor 0.5, with the
same lower bound. This gives large inverse-response steps to locally saturated
sites while preventing one noisy secant estimate from producing an unstable
jump.

The convergence test remains global for each Cartesian component,

$$
\|\langle m_a\rangle-m_a^{\rm target}\|_2<10^{-4},
\qquad a=x,y,z,
$$

while the adaptive $\alpha_i$ and failure reports remain site resolved.

## Parameters and execution

| Quantity | Value |
|---|---:|
| Lattice | $16\times16$, periodic |
| Sites | 256 |
| Filling | half filling (`fil = 0.5`) |
| Nearest-neighbor hopping | $t_{\rm nn}/U=0.25$ |
| Longitudinal mobility | $\Gamma_\parallel=0.1$ |
| Transverse mobility | $\Gamma_\perp=0.3$ |
| Timestep | $\Delta t=0.05$ |
| Field tolerance | $10^{-4}$ per global Cartesian $L^2$ residual |
| Maximum field iterations | 3000 |
| Initial/local $\alpha_i$ bounds | $0.5\le\alpha_i\le64$ |
| CUDA seed | 2025 |

The code reads its run configuration from environment variables:

```bash
export SDW_TELEC=0.01
export SDW_TFLUC=0.01
export SDW_NRUNS=10000
export SDW_OUTPUT_DIR=/path/to/output
julia Neel16_direct_M_reflected.jl
```

To reproduce the higher-temperature comparison, set both temperatures to
`0.03`. A Slurm launcher is included:

```bash
SDW_TELEC=0.01 SDW_TFLUC=0.01 sbatch run_reflective_sde.slurm
SDW_TELEC=0.03 SDW_TFLUC=0.03 sbatch run_reflective_sde.slurm
```

The implementation writes component snapshots, field-solver diagnostics,
reflection counts, per-stage iteration counts, and site-resolved failure
histories. The result images in this directory are representative snapshots;
the raw trajectories remain too large for this repository.

## Interpretation and next checks

The present comparison supports a fluctuation-inclusive finite-size crossover:
the $L=16$ trajectory is Néel ordered at $T=0.01$ and visually disordered at
$T=0.03$ when the electronic and stochastic temperatures are matched. It does
not yet locate a thermodynamic critical temperature. The next quantitative
analysis should measure the staggered structure factor, correlation length,
autocorrelation time, and results across independent seeds and several lattice
sizes. Timestep refinement is also required when reflections become frequent.
