# Conserving external GF2 response and two-particle diagnostics

The stable native GF2 implementation is unchanged. The new
`python/gf2_response.py` audits its finite, orthonormal, collinear checkpoints
and computes an external conserving response. This is the same stable-v1.0.0
line used for the spatial cavity work; no symmetry/RSGDF development solver is
involved. The application and execution evidence live in the separate
[Hall repository](https://github.com/Chenan-Phys/green-cavity-quantum-hall).

## Electronic functional and response

For Hermitian DF vertices, `v_abcd=sum_Q L_Qab L_Qcd`. The orbital-tensor reference
contracts the direct and exchange second-order terms independently of the native
DF code. In the code's notation, the direct part is
`v_ibac G^s_bd(tau) v_djef G^r_ce(tau) G^r_fa(beta-tau)`, summed over internal spin r;
the same-spin exchange subtracts the corresponding `v_dfej` term. A restricted
single-spin checkpoint has weight two in the direct spin sum.

If this trilinear functional is F, all electron lines are varied:

\[
\delta\Sigma^{(2)}_\nu(\tau)=
F[\delta g_\nu,G,G^-]+F[G,\delta g_\nu,G^-]
+e^{i\nu\tau}F[G,G,\delta g_\nu^-].
\]

Hartree and Fock derivatives are included as well. `G^-` denotes the reverse
imaginary-time argument, not complex conjugation. The transfer convention is
the one documented in [conserving_response.md](conserving_response.md).
The existing two-leg IR representation and matrix-free GMRES solver are reused.
There is no screening-only approximation or frozen self-energy in the conserving
variant. `fixed_W` is deliberately unavailable for GF2.

For larger translation-invariant strips, see
[the exact block representation](translation_response.md). The same functional
can omit its forbidden momentum elements, and spin-resolved one-body terms are
accepted. Numerical HF preconditioning changes the linear solver, not its kernel.

The class rejects a checkpoint unless reconstruction of Dyson, Hartree/Fock and
the full cubic GF2 self-energy meets a 1e-7 absolute residual bound. Its internal
`static_derivative_check` takes a variation already expressed on the response
frequency grid; the application audit shows its construction. Public `solve`
returns the same diagnostics and spin-resolved density variation as the GW class.

## Weak photons

The inherited photon-coefficient helper integrates the conserving susceptibility
of the native GF2 state and adds the quadratic and coherent-current contributions.
J and K come from the same application Hamiltonian as in the GW comparison.
This supplies a second-order weak-photon free-energy coefficient around electronic
GF2. It does not implement a coupled finite-strength QED-GF2 C++ loop. The existing
native spatial photon engine still accepts HF and GW only.

## Independent checks

The Hall evidence includes random complex spin-orbital antisymmetrized contractions,
complex basis covariance, a polarized one-electron atomic cancellation, native
complex covariance, functional finite differences, dynamic conserved-number
response, exact free Lehmann response over all 71 nonnegative bosonic nodes,
a free photon coefficient, an interacting open-plaquette static gauge test,
restricted/unrestricted spin-weight equivalence and native fixed-N density finite
differences. The latter has errors 3.43e-8 and 8.58e-9 at steps 0.001 and 0.0005.
The two-step decline checks the expected central-difference limit. A native
energy stopping criterion alone was insufficient for one trial checkpoint;
30 tighter iterations reduced its static self-energy residual from 6.2e-7 to
2.1e-9. That corrected checkpoint is separately identified.

## Spin and number diagnostics

`python/two_particle_diagnostics.py` implements a fixed-G functional-derivative
diagnostic using the strategy of
[Pokhilko et al., JCP 155, 024119 (2021)](https://arxiv.org/html/2104.10308v1).
It includes the disconnected correlated-density product and the approximate
two-particle cumulant. GF2 uses antisymmetrized vertices; the application verifies
the reduction against finite perturbations of the full spin-orbital Phi_2
functional and checks its correlation-energy contraction and quadrature.

GW uses an explicitly stated exchanged-annihilator operator representation,
validated against full pair-space Phi derivatives. Its cumulant lacks fermionic
antisymmetry, so equivalent operator representations can produce different
diagnostics. The returned GW value must not be treated as a unique spin projection.
Neither method's thermodynamic approximate cumulant is interchangeable with the
full conserving BSE response. Truncations can violate exact relations between
one- and two-particle observables. Negative number variance flags a two-particle
representability failure; it is not repaired by projecting the answer to zero.

The application found such a negative variance for a broad equal-spin GF2 branch.
Its converged one-particle density and energy are retained as diagnostics, but it
is not identified as the exact singlet or used for a physical cavity phase claim.

`python/collinear_response.py` applies independent up/down one-body probes to the
same kernel. In the broad-branch test, the conserving static longitudinal spin
susceptibility is positive, while its dynamic global-spin response vanishes to
numerical precision. Thus the negative fixed-G Phi cumulant variance does not
alone demonstrate a conserving-response instability. These estimators are saved
separately; no inconsistent variance is used to project or modify the response.
