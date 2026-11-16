# Conserving response of the finite electronic GW solution

`python/conserving_response.py` is a matrix-free reference implementation of
the external linear response of a converged finite-system electronic scGW
checkpoint. It adds a response calculation around the stable-based solution.
It does not change the C++ ground-state solver or implement a fully
vertex-corrected electron–photon self-consistency loop.

The formulation follows the functional-derivative conserving-response
construction of [Baym and Kadanoff](https://doi.org/10.1103/PhysRev.124.287).
The distinction between an internal GG polarization and a physical response,
including the derivative of screening, is made explicitly in
[Kutepov and Kotliar](https://doi.org/10.1103/PhysRevB.96.035108).

The continuum-strip application now explicitly orders the DF contractions as
`L G`, then screening, then the second vertex. This is the same GW derivative;
it avoids a costly auxiliary-pair outer-product path selected by NumPy's greedy
four-operand optimizer. Random complex direct contractions agree within 8e-14,
and the 995-check main response suite was rerun successfully. On the 8-orbital,
48-auxiliary strip, engine construction fell from about 59 s to 0.48 s and one
measured response action took 0.87 s on Pauli. These are reference-Python timings,
not native GREEN solver benchmarks.

## General Coulomb weak-photon reference

`python/df_photon_reference.py` supplies a small fixed-spin-sector ED reference
for general complex Hermitian DF vertices, beyond density-density interactions.
Normal ordering is explicit:

\[
H_e=d\Gamma(h)+\frac12\sum_Q\{d\Gamma(L_Q)^2-d\Gamma(L_Q^2)\}+c.
\]

For one mode `eta Q J + eta² Q² K`, define
`u=-(H_e-E_0+omega)^(-1) J|0>` and
`v=-(H_e-E_0)_perp^(-1) [K|0>+J u]_perp`.
Then `C=<K>+<0|J|u>` and
`delta<O>/eta²=<u|O|u>-<u|u><O>+2 Re <0|O|v>` for Hermitian O.
The code evaluates the full density matrix, and checks fixed particle number
without projecting its trace. It includes a separate photon-Fock diagonalization
for the eta-to-zero energy and density limits. A nondegenerate electronic ground
state and positive mode frequency are required; this is a zero-temperature
reference. Minimal Sz includes higher-total-spin states, so the returned S²
diagnostic must be checked before comparison with a restricted GW solution.
An explicit `singlet=True` comparison adds `100 S²` to select S=0 in the minimal
Sz space, with S² built from signed fermionic spin-raising operations. The returned
spin expectation must be zero. Spin-scalar current and density operators preserve
that subspace, so the penalty does not change its physical resolvents. A singlet
reference must not be presented as the unrestricted ground state when another
total-spin sector is lower.

## Equations differentiated

For a one-body perturbation u O exp(-i nu tau), solve the linearized Dyson
equation for delta G(iw+i nu,iw):

\[
\delta G_\nu(iw)=G(iw+i\nu)
 [O+\delta\Sigma_\nu(iw)]G(iw).
\]

The implementation differentiates every electronic GW contribution:

\[
\delta\Sigma=\delta\Sigma_H+\delta\Sigma_x
 -L\,\delta G\,L\,W_c-L\,G\,L\,\delta W,
\qquad
\delta W(i\Omega+i\nu,i\Omega)
=W(i\Omega+i\nu)\delta\Pi_\nu(i\Omega)W(i\Omega).
\]

All repeated orbital, auxiliary and spin indices are contracted. Delta Pi
contains variations of both electron lines. W includes the instantaneous
Coulomb identity; Wc excludes it, because Fock exchange is already differentiated
separately. Hartree uses the total spin density and exchange uses each spin
channel. The linear solve uses GMRES with matrix-free actions; it does not
construct a full four-orbital/two-frequency vertex tensor.

With the convention
delta G(t1,t2)=exp(-i nu t1) delta g(t1-t2), native Pi=-GG(beta-tau) gives

\[
\delta\Pi_{QR,\nu}(\tau)=-\sum_s\operatorname{Tr}
 [L_Q\delta g_s(\tau)L_R G_s(\beta-\tau)
 +e^{i\nu\tau}L_QG_s(\tau)L_R\delta g_s(\beta-\tau)].
\]

For ns=1 the spin sum has weight two. The density variation is
delta rho_s=-delta g_s(beta), and the positive connected susceptibility is
chi_OO=-sum_s Tr(O delta rho_s), with the same restricted spin weight. No
Hermitian projection is applied to an individual frequency-transfer variation:
that variation need not be Hermitian. No total-charge projection, operator
centering or imposed Ward vertex is used in the solution.

## Frequency-transfer representation

Using the ground-state IR span alone fails at moderate transfer frequency.
Delta G has singularities associated with both iw and iw+i nu. The response
therefore uses the joint span U_l(iw), U_l(iw+i nu), or equivalently
U_l(tau), exp(i nu tau)U_l(tau). The bosonic variation uses the corresponding
two-leg span. Sample frequencies are the union of the original Matsubara
indices and those shifted by the transfer index. Both legs are additionally
oversampled within 32 Matsubara indices of their respective low-frequency
centers. This removes otherwise weakly constrained interpolation directions.
Extra Gauss–Legendre time
points supplement the native fermionic/bosonic time grids.

The joint span is overcomplete. Direct pseudoinversion produces cancellation
errors. Weighted time-space SVD removes near-null directions before forming a
frequency inverse. The initial cutoff is 1e-10; the weighted frequency inverse
uses the same cutoff. Baseline G, Sigma and W are evaluated using the installed
irbasis database, with their fitted reconstruction errors checked against native
data. The shared basis functions agree with the native grid; a few extra native
functions beyond the packaged database are negligible for the tested models.
Runtime code uses scalar basis-index evaluation to avoid a broadcasting defect
in the installed irbasis vector-index path.

## Controls and exact identities

`conserving` differentiates the full GW functional. `fixed_W` includes Hartree,
exchange and delta G Wc, but omits G delta W. `bubble` freezes all self-energies.
`Hartree` varies only Hartree around the SAME correlated GW propagator; it is
the internal bubble plus Coulomb screening, not a separate Hartree ground-state
calculation. `projected_rpa` computes that internal response independently by
projecting the native polarization and screened Coulomb matrices.

For a total-number source at nonzero nu, the exact response is zero. A useful
independent solution check is

\[
\delta G_{N,\nu}(iw)=[G(iw)-G(iw+i\nu)]/(i\nu).
\]

This solution is checked AFTER the GMRES solve, not substituted into it. In
this special channel delta Pi and delta W cancel, so the number test alone
cannot validate the G delta W term. A central finite difference of the full GW
self-energy functional tests that term. Finite differences of freshly converged
native GREEN calculations test the external static density response as well.

For a uniform pure-gradient vector potential on an open density-interaction
lattice, the static paramagnetic susceptibility must cancel the diamagnetic
curvature: chi_JJ(0)=2 Tr(rho K), using the same approximation's density. This
is an additional spatial gauge control. It is not a proof of complete
electromagnetic gauge equivalence for every truncated device basis.

## Weak photon coupling

For H_int=eta Q O+eta² Q² K and a cold, gapped electronic reference, the
second-order ground-state energy coefficient is

\[
\Delta E=\eta^2\left[
 \langle K\rangle-\frac{\langle O\rangle^2}{\omega}
 -\frac{1}{2\beta}\sum_\nu C_0(i\nu)\chi_{OO}(i\nu)
 \right]+O(\eta^4),\qquad
C_0(i\nu)=\frac{2\omega}{\nu^2+\omega^2}.
\]

The connected photon Q-variance coefficient is

\[
\delta\langle\delta Q^2\rangle/\eta^2
=-2\langle K\rangle/\omega
 +\beta^{-1}\sum_\nu C_0(i\nu)^2\chi_{OO}(i\nu).
\]

At finite beta the response trace computes a free-energy coefficient; the
zero-temperature interpretation requires the cold/gap check. The initial
tests use beta*t=60 and omega/t>=0.5. Susceptibilities are solved on the native
bosonic IR frequency nodes, and the decaying products with C0 are summed with
the original IR transform. Independent fixed-N electronic Lehmann spectra give
the exact second-order coefficient without a photon cutoff. Finite-photon ED
checks the approach to that coefficient at small eta.

The conserving response and the original internal-GW response define different
weak-cavity estimates. Replacing one response in a fully coupled feedback loop
would require a consistent treatment of its self-energy and energy functional.
This module does not claim to have done that. Nor does it supply the cavity-
induced electronic density correction at order eta²: its delta rho is the
linear response to the external classical source.

## Supported initial domain

Finite orthonormal Gamma-point electronic GW, Hermitian Coulomb DF vertices,
one or two scalar spin channels, no antiunitary reduction, and the original
native IR grid/checkpoint convention. Initial integration tests are four-site,
two-electron gapped models. The static response is at fixed chemical potential;
for a metallic fixed-N application, an explicit chemical-potential constraint
is needed. Dynamic total-charge conservation is tested without that constraint.

Heavy numerical work must run on an allocated remote compute node. Hall-specific
profiles, native finite-difference runs, independent ED comparisons, figures and
measured validation status belong to the separate cavity-Hall project.

## Initial measured validation

The Hall project records 995 integration checks, 56 representation/spin controls
and 144 conserved-number integral checks, all passing on allocated Pauli nodes.
The full self-energy functional derivative matches a central finite difference
to 1.6e-11 relative error; freezing delta W gives a 0.69% derivative discrepancy
in that test. The static density response matches freshly converged native
GREEN finite differences to 4.9e-8 at source step 1e-3 and 1.2e-8 at step 5e-4.

In the four-site, two-electron interacting model (Coulomb scale 0.3t, photon
omega=t), the exact weak-cavity energy coefficient is 0.1734273t. The internal
GW response gives 0.1946951t (12.26% error); the conserving external response
gives 0.1733498t (0.045% error). The same electronic G and density enter both
approximations. The uniform static gauge curvature is restored to about
3.4e-11t. The original conserved-number energy and photon-variance diagnostics,
evaluated as weak-cavity estimates at eta=0.04, have residuals of order 1e-13.

These results validate this external-response prototype on the specified small
models. They do not alter the original C++ photon feedback loop or establish
device-scale accuracy. See the Hall report and machine-readable summaries.
