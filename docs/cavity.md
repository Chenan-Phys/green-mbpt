# Molecular cavity extension on stable GREEN

Branch: `cavity-general`; base: `d593a9c970194dea5855ca38e71f9cd6ac550cbf`.
This is an implementation under validation, not yet a validated production release.

## Hamiltonian and input

Use the fixed-nuclei length-gauge single-mode Pauli–Fierz Hamiltonian in atomic units,

\[
H=H_e+\omega b^\dagger b-\sqrt{\omega/2}\lambda\mu(b+b^\dagger)+\lambda^2\mu^2/2.
\]

The photon zero point is omitted. The projected electron dipole matrix is
`d=-<e.r>`. The separate second-moment matrix is `<(e.r)^2>`; it must not be replaced by `d S^-1 d`.
Input `QED` datasets: `schema=1`, `molecular=1`, `omega_hartree`, `lambda_au`,
`nuclear_dipole`, `dipole[nao,nao]`, `second_moment[nao,nao]` (real float64).
The same AO basis must be used for these operators, GREEN matrices and fitted Coulomb factors.
The first implementation supports scalar HF and joint GW, CPU, one MPI rank,
one molecular k point. Unsupported GF2/GPU/periodic/spinor combinations fail explicitly.
Absence of the QED group retains the original electronic calculation.

## Joint screening

Append `lambda*d` to the Coulomb fitted vertices. Its bare auxiliary interaction is
`b_ph(i nu)=nu^2/(nu^2+omega^2)` after integrating out the free photon. The Coulomb
auxiliary entries are one. This combines the instantaneous DSE with the retarded photon
exchange; its static cancellation is retained.

The existing full-matrix bubble gives `P`. With the diagonal positive square root `R` of
the bare auxiliary interaction, compute

`W = R (I - R P R)^-1 R`.

Off-diagonal Coulomb–dipole blocks are included. The dynamical self-energy contracts
`W-I` with G and the fitted vertices. `I` here is the **instantaneous** Coulomb-plus-DSE
reference. Subtracting the frequency-dependent bare interaction would wrongly remove
bare photon exchange. The electronic density and full matrix G are updated by the
native self-consistency driver.

The coherent displacement cancels the direct dipole Hartree term. The static addition
for each spin is `lambda^2*r2/2 - lambda^2*d*gamma_spin*d`. The total dipole including
ionic charges determines the reported coherent displacement.

## Energy estimator: derivation to validate

Let `chi = P + P W P` in auxiliary space, including the coupling-scaled dipole vertex.
Here `P`/`chi` use the negative density-response convention, and the bosonic
Green function is `D(tau)=-<T q(tau) q(0)>_connected`, with `q=b+b†`.
The connected photon correction is

`delta D(i nu) = 2*omega^3*chi_ph,ph(i nu)/(nu^2+omega^2)^2`.
The connected coordinate variance is `1+2*n_B-delta D(tau=0)`.

Relative to the native electronic Galitskii–Migdal estimator, the proposed correction is

\[
\delta E=\frac{\lambda^2}{4}\operatorname{Tr}(r^2\gamma)
+\omega n_B(\omega)
+\frac1\beta\sum_\nu
\frac{\omega^2\nu^2}{(\nu^2+\omega^2)^2}\chi_{ph,ph}(i\nu).
\]

The first term restores the full weight of a one-body DSE contribution stored in
`Sigma_inf`. The last two terms are `E_photon + E_bilinear/2`: native GM already counts
half the bilinear interaction. The frequency contraction follows from the connected
photon propagator and bosonic virial identity. The free-photon contribution is handled
analytically to avoid fitting its equal-time constant from sparse frequencies.
This expression must pass weak-coupling, direct-Hamiltonian and grid checks before
charging energies are interpreted. Approximation errors remain distinct from these checks.

The native `Energy_HF` and `Energy_2b` iteration fields incorporate these corrections;
`Energy_total_QED` and photon diagnostics are also saved. Connected bosonic changes
participate in the convergence criterion. The current spin-unrestricted driver fixes
total electron count by default; inspect individual spin counts in charged calculations.
The optional `QED/fixed_spin/target[2]` dataset constrains the two mean spin
populations through separate chemical-potential searches. It requires ns=2 and
constant density, and the populations must sum to the input electron count.
The chemical potentials are Lagrange multipliers, not external magnetic fields;
the original electronic Hamiltonian and energy estimator are unchanged.
This constrains mean populations at finite temperature, rather than implementing
an exact canonical ensemble. `Mu_spin` and `Target_spin_electrons` are saved.

## Validation and limitations

All compilation and scientific tests run on GREEN_workstation. The build script uses
the installed release's dependency source caches in a separate build and install prefix.
It does not rebuild or overwrite the installed stable executable.

Required checks: electronic zero-coupling recovery, correct coherent HF expectation,
independent finite electron–photon diagonalization, even coupling parity, grid/temperature
convergence, charge counts, and origin sensitivity. Internal bubble screening is not
automatically a physical optical response. Gauge/origin sensitivity of an approximate
bare-vertex screening treatment must be measured, not assumed absent.

### Results obtained on 2026-10-06

The optimized build passes 26/26 native tests. H2/STO-3G at 0.74 Angstrom,
omega 2 eV and lambda 0.005 au gives a GW cavity energy shift about 0.55257 meV,
compared with 0.53557 meV from finite electron-photon CI. Photon cutoffs 8/16
agree within 1e-9 Ha. The independent coherent HF expectation agrees at numerical
precision. Coupling parity and zero-coupling recovery pass. H2+/H2- checks recover
the expected spin occupations and validate the unrestricted HF energy estimator.
The optional fixed-spin version passes the same references and 26 native tests.

**Unresolved:** shifting the dipole origin by 1 Angstrom changes the H2 GW
energy by 1.0718e-6 Ha (0.02917 meV), while HF is invariant. This is appreciable
relative to the benchmark cavity signal. Origin/number-response and vertex
analysis must precede physical interpretation of small application GW shifts.
These results qualify an experimental implementation, not a production release.

`cavity_run.py` records input, binary, core-source and BLAS-runtime hashes;
explicit checkpoint restarts preserve existing HDF5 data. It defaults to a
1e-8 Ha pilot threshold, with tighter tolerances available. The installed
pthread LP64 OpenBLAS variant is used only for the native child process;
the default system BLAS is single threaded. `cavity_collect.py` exports compact
manifests and per-method statuses. Scientific scripts must run on GREEN_workstation.

The sodium-cluster and TTF-TCNE applications, their launchers, and their separate
repositories were permanently removed on 2026-10-06 at the user's request.
The shared solver and independent small-system validation remain. Beta 1000 Ha^-1
is a finite-temperature pilot; an internal-energy difference is not automatically
a free-energy difference. Historical application examples below document checks
that informed the shared implementation, rather than active application runs.

The acceptance check also requires an unnormalized overlap-weighted density
residual below 1e-6 electron and agreement of the reported static HF energy with
an independent same-factor PySCF evaluation below 1e-7 Ha. This check applies
to the HF component of GW as well. A sodium cation DIIS run passed successive
energy/density changes but failed the functional-energy check by 1.1e-5 Ha;
it was excluded from charging differences. Linear
self-energy mixing stabilizes the initial TTF GW control after DIIS oscillation.
Mixing type/weight are explicit runner/campaign options. Each case has a
nonblocking file lock plus a check for legacy native writers in its directory.
The serial queue checks memory and free disk space before starting a case.
It retains the existing checkpoint histories.
Its disk check reserves the full bounded 120-iteration G/self-energy history
with 25% overhead and a 2 GiB margin, in addition to 60 GiB remaining free;
the time-grid size is read from the installed grid. An insufficient-space
result stops the campaign before launching another case.

`cavity_thermal_hf.py` independently solves the finite-temperature coherent
HF equations with the same saved factors, reporting entropy, internal energy,
free-energy functional, spin counts and a Fock/density commutator. It bypasses
the one-electron shortcut and builds full custom potentials to avoid retaining
an earlier cavity correction in incremental updates. The one-electron H2+
reference passes; the initial Na20 cation reference does not converge and is
recorded as failed rather than substituted for a native result.

### Density convergence, initial guesses and basis controls

The current native build adds the physical density criterion directly to the
SC loop: the unnormalized Frobenius norm of `S^1/2 delta_gamma_spin S^1/2`
must be below 1e-6 electron, together with the energy and bosonic criteria.
The first iteration cannot pass this density criterion. Restricted densities
include their factor of two. The new install prefix is
`$HOME/green/install-qed/cavity-general-density`; its build manifest records
the executable hash and six LF-normalized core-source hashes. Runners reject
an executable/current-core mismatch. Earlier results retain their earlier
binary and source hashes. The rebuilt solver passes 26/26 native regressions
and the charged H2 benchmarks.

`cavity_warm_start.py` preserves the source solution and seeds a fresh target
coupling point with its last two GW iterations. The system, basis representation,
mode and spin sector must match; the target has its own Hamiltonian and must
converge independently. Its charged H2 test agrees with an independent cold
solution within 9.4e-11 Ha. `cavity_hf_warm_start.py` instead prepares a fresh
input using an accepted independent HF starting Fock, with physical operators
and factors unchanged; it passes the H2 check.

`cavity_lowdin.py SOURCE TARGET` makes a fresh full symmetric `S^-1/2`
representation, transforming H, S, the starting Fock, dipole, actual second
moment and every Coulomb factor together. It discards no basis functions.
Original AO factors and `native_to_ao` remain available for independent energy
evaluation and real-space densities. The analyzer saves physical AO density
and uses the original symmetric Lowdin populations. Cold transformed H2 HF/GW
energies agree within 1.2e-15 Ha, and AO densities within 2.7e-15. This is a
conditioning control, not a change of electronic basis or a cure for soft
orbital modes. Use the original physical specification JSON with the prepared
converted directory; the exporter rejects a misleading `native_basis=lowdin`
request without an actual conversion.

Thermal HF references report an overlap-invariant Fock/density commutator.
Their default orbital gradient tolerance is 1e-7 Ha; an explicit pilot setting
up to 1e-6 Ha is available and recorded, while the commutator gate remains
1e-6 Ha. Native energy/density acceptance is still required after using a
reference as a starting guess. A fixed mean spin sector does not imply spin purity.

The exact H2 photon-moment test has converged at cutoffs 8/16. Its induced
connected coordinate variance is 3.4518e-5 versus 2.1448e-5 in GW: about 38%
smaller in GW, despite the closer energy shift. This compares fitted GW with
unfitted finite CI. `Photon_energy_correction` means `E_photon+E_bilinear/2`,
and must not be interpreted as photon occupation energy.

### Orbital stability and chemical-potential diagnostics

`cavity_hf_stability.py` checks accepted, gapped UHF references using the
internal orbital Hessian with the PF exchange response. It records the
PySCF -1e-5 Ha negative-eigenvalue criterion and can provide an unstable-mode
density for a fresh relaxation. Passing this test is neither global optimality
nor spin purity. It exposed Na8 cation saddles; successive repairs allowed all
six native Na8 HF controls to pass. Restricted neutral states have not been
tested for unrestricted instabilities.

The exporter now explicitly Hermitizes full-basis transformed operators and
factors after bounding their asymmetry as relative roundoff. The separate
`cavity_hermitian_input.py` makes a fresh corrected copy with recorded input
hashes and correction norms. This repairs a Na20 second-moment asymmetry of
1.87e-10 without relaxing the native gate; the original input is preserved.

`cavity_run.py --fixed-chemical-potential --restart-unconverged` exposes stable
GREEN's existing constant-mu mode for initialized checkpoints. G and Sigma
still iterate; the mean particle number must pass the unchanged postprocessing
gate. This mode rejects fixed-spin inputs and cannot cold-start from mu=0.
Its H2 comparison passes within 3.47e-9 Ha and 5.26e-9 in AO density. A Na8
diagnostic converges in energy but misses N by 2.73e-5 and is excluded. Sodium
production controls therefore retain chemical-potential searches. Provenance
and caches distinguish these ensembles. JSON acceptance flags are normalized
to Python booleans so rejected number checks can be saved reliably.

`cavity_lossless_compress.py` creates a gzip/shuffle-compressed copy of dormant
task result files. Every dataset byte, datatype, shape, maximum shape and
attribute enters a storage-independent SHA-256 comparison. All iterations are
retained. The default retains the original; `--replace-original` requires
explicit authorization for the named remote files. A fresh copied H2 control
requires verified gzip filters, identical diagnostics and a native restart.
For this small validation copy only, compression may increase its size; the
utility normally retains originals when repacking gives no size reduction.
Inputs and integral caches are outside this utility's scope.

### Analytic limits and physical readiness

The 2026-10-06 workstation validation adds 28 passing analytic checks in 12
synthetic two-orbital cases (`cavity_limits.py`): zero coupling, the analytic
O(lambda^2) energy and connected photon-variance coefficients at three couplings,
coupling parity, orbital rotations, the full projected second-moment correction,
a dipole proportional to conserved particle number, a free thermal photon,
restricted/unrestricted closed-shell equivalence, and IR grids 1e5/1e6.
The grid energy difference is 7.34e-12 Ha. These Hamiltonians use zero electronic
Coulomb factors and explicitly modified one-body operators; they are mathematical
validation models, not H2 predictions. Independent HF and finite CI references
read the saved one-body Hamiltonian and nuclear energy. The optional
`exact_reference(..., use_saved_df=True)` reconstructs the same saved fitted
Coulomb Hamiltonian as GREEN, avoiding an unfitted-versus-fitted comparison.

`cavity_oracle.py` independently contracts a saved G with explicit NumPy tensor
indices, solves W=(I-BP)^(-1)B directly, and checks the dynamic electronic
self-energy and connected photon diagnostics. Four converged H2/model cases
pass; the largest self-energy difference is 3.07e-12 Ha. Saved self-energies are
mixed iteration quantities, so this is a converged fixed-point comparison.
Six deliberately invalid/unsupported native inputs fail with their intended
messages (`cavity_invalid_validate.py`). These checks supplement the previously
passing 26 native CTests; the C++ core and executable are unchanged.

The runner now verifies the current executable/build identity even when returning
a cached result. Checkpoint input digests must match, including explicit restarts;
temperature and grid must match. Cached core/binary identities and ensemble must
also match. Unknown checkpoint provenance requires a fresh case. A warm-start
record is marked `requires_reconvergence`; legacy `initial_guess` records receive
the same treatment. Neither can be returned as a converged target without a
native continuation, even if the lambda change is smaller than postprocessing
tolerances. Twelve workflow regressions pass, including two actual native seed
continuations and preservation of the source checkpoints.

The physical origin-invariance gate still fails. A fresh H2 comparison with
energy tolerance 1e-12 Ha, IR grid 1e6, and identical saved DF factors gives an
origin energy change of 1.0718154e-6 Ha in GW, versus below 2.5e-15 Ha in HF
and finite electron-photon CI. The GW cavity shift differs from matched CI by
3.19%, and its induced connected photon coordinate variance is 37.87% smaller.
The photon cutoffs 8/16 converge. Tightening the numerical setup and matching
the Hamiltonians does not remove these approximation errors.

The internal bare-bubble/ring response has a nonzero finite-frequency response
to conserved N in interacting H2. This is evidence of a response-vertex problem,
not a failure of macroscopic particle conservation by self-consistent GW.
Published [molecular QED-GW theory](https://arxiv.org/html/2609.00594v2) explicitly
discusses the lack of translation invariance of individual ring correlation
energies. The numerical observations are consistent with that limitation;
they do not prove that every aspect of the implementation is correct.
Centering a dipole by convention does not constitute passing the physical gate.
A consistent response-vertex/coherent-transformation method extension needs its
own derivation and benchmarks before small application GW signals are claimed.

`cavity_reference_audit.py --require-origin-invariance` saves evidence then fails
if the physical gate fails. `cavity_validation_status.py` combines the limits,
workflow, invalid-input, oracle and matched-reference JSON reports; with
`--require-physical` it returns nonzero for failed physical readiness even when
every implementation check passes. Sodium production GW was checkpointed and
paused while this limitation is assessed. No accepted charged GW difference
is inferred from unconverged checkpoints.
