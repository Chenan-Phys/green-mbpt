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
The connected photon correction is

`delta D(i nu) = 2*omega^3*chi_ph,ph(i nu)/(nu^2+omega^2)^2`.

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
total electron count; inspect individual spin counts in charged calculations.

## Validation and limitations

All compilation and scientific tests run on GREEN_workstation. The build script uses
the installed release's dependency source caches in a separate build and install prefix.
It does not rebuild or overwrite the installed stable executable.

Required checks: electronic zero-coupling recovery, correct coherent HF expectation,
independent finite electron–photon diagonalization, even coupling parity, grid/temperature
convergence, charge counts, and origin sensitivity. Internal bubble screening is not
automatically a physical optical response. Gauge/origin sensitivity of an approximate
bare-vertex screening treatment must be measured, not assumed absent.
