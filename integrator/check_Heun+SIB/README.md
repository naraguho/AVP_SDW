# Combined longitudinal-Heun + transverse-SIB verification

This folder verifies the numerical composition used for a soft vector `m = M e`. It answers a question that the standalone SIB test cannot: when a fluctuating scalar amplitude is recombined with an independently integrated orientation, does the complete vector inherit the expected stochastic accuracy?

In the manuscript, this corresponds to composing the amplitude equation Eq. (62)/Eq. (D11), integrated by Heun, with the orientation equation Eq. (63)/Eq. (D13), integrated by SIB as specified in Appendix D.3. See the [equation-level manuscript crosswalk](../MANUSCRIPT_CROSSWALK.md).

![Combined Heun-SIB convergence tests](results/heun_sib_verification.png)

The plotted quantity is the complete physical vector `m = M e`. The figure makes clear that the transverse SIB sector limits the finite-temperature strong order of the combined method. The plotting source is [`plot_results.jl`](plot_results.jl).

## Manufactured model

The test uses two exactly characterized channels:

```math
dM=-\lambda(M-\bar M)dt+\sigma_M dW_\parallel,
```

```math
d\mathbf e=\mathbf e\times(-\mathbf B)dt-
\sqrt{2D}\,\mathbf e\times\circ d\mathbf W,\qquad |\mathbf e|=1,
```

with `m = M e`. The amplitude is an additive-noise Ornstein-Uhlenbeck process advanced by stochastic Heun. The direction is constant-field precession plus isotropic Stratonovich rotational diffusion advanced by SIB/Cayley. The channels are independent, permitting exact amplitude moments and factorized exact weak moments of the full vector.

## Implementation

`amplitude_heun` performs an Euler predictor followed by a trapezoidal drift correction. The same scalar Brownian increment appears in both stages.

`direction_sib` performs the two implicit-midpoint rotations. In this manufactured problem the field and noise coefficient are state independent, so the two SIB rotation vectors coincide; the two-stage structure remains explicit in the source.

`hybrid_step` advances both channels and reconstructs `m` only after their individual updates. It never applies SIB to the amplitude and never renormalizes the full soft vector.

## Demonstrations

### Deterministic test

Both exact solutions are known. Heun is second order for the amplitude and the Cayley/SIB constant-field step is second order for the direction. Therefore the complete vector should be second order.

### Strong finite-temperature test

The coarse and fine simulations use the same Brownian path: coarse Wiener increments are sums of fine-grid increments. The RMS path errors are measured separately for the amplitude, direction, and full vector. The amplitude has strong order about one for additive noise, whereas SIB has strong order one half for the multiplicative orientation noise. Consequently the full vector is limited to strong order one half.

### Weak finite-temperature test

The exact OU mean and variance are compared with the discrete Heun moments. The orientation mean is evaluated through the exact radial expectation of one Cayley step. Because the channels are independent, the mean of the full vector factorizes into the product of the amplitude and orientation means. The expected amplitude-moment order is two, while the orientation and full-vector weak order is one.

### Geometry test

A 100000-step trajectory checks both unit orientation length and the identity `|M e| = |M|`. This catches accidental normalization of the soft vector or drift of the direction norm.

## Reproduction

```bash
julia heun_sib_combined_verification.jl --quick
julia heun_sib_combined_verification.jl
julia plot_results.jl
```

## Full-run results

| Quantity | Observed order | Expected order |
|---|---:|---:|
| deterministic amplitude | 2.027967 | 2 |
| deterministic direction | 1.998292 | 2 |
| deterministic full vector | 1.998443 | 2 |
| strong amplitude | 1.011903 | about 1 |
| strong direction | 0.520452 | 0.5 |
| strong full vector | 0.521103 | 0.5 |
| weak amplitude mean | 2.019259 | 2 |
| weak amplitude second moment | 2.018954 | 2 |
| weak direction mean | 0.988901 | 1 |
| weak full-vector mean | 0.979567 | 1 |

The maximum direction-norm error was 2.64 × 10⁻¹⁴, and the maximum error in `|M e| = |M|` was 2.66 × 10⁻¹⁴.

These deviations are small finite-range/statistical effects. The full vector shows the limiting orders predicted by the two component methods: deterministic order two, strong order one-half, and weak order one.

## What this test does not establish

The manufactured field is constant and the amplitude/direction channels are independent. Therefore this test validates the split numerical composition but does not test the state-dependent electronic field, chemical-potential solve, coupling between longitudinal and transverse coefficients, behavior near zero amplitude, or the equilibrium distribution of the production SDW model. Those claims require the actual-code refinement protocol described in the [parent integrator methodology](../).
