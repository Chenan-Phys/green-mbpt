# Conserving response for translation-invariant material strips

The native stable-based GF2/GW executable and its electronic functionals are
unchanged. These Python representations make the same external response
practical for larger strips. The application remains in the separate Hall
repository. They do not use the developing GREEN symmetry or RSGDF solvers.

`GWResponse` and `GF2Response` now accept a common `(n,n)` one-body Hamiltonian
or collinear spin-resolved `(ns,n,n)` terms. This permits a physical Zeeman term.
`low_energy_dyson.py` provides a low-frequency Matsubara quasiparticle estimate
with fit-window, scattering and metric diagnostics. It is not unique analytic
continuation or a replacement for a real-frequency spectral analysis.

## Exact invariant subspace

For orbitals ordered by guiding center and then Landau index, a y-invariant
strip and a q_y=0 probe preserve each guiding-center block. Unwrapped momentum
is conserved. There is no artificial periodic wrap at the transverse basis
boundary. A probe mixing these blocks is rejected.

`translation_gf2.MomentumGF2` evaluates the unchanged cubic direct/exchange
functional using the allowed internal momentum triplets. `GF2Response` selects
it only when `momentum_block_size` is supplied. Its derivative still varies all
three electron lines. The default full contraction remains available and now
chunks its time intermediates for larger orbital spaces. GF2 no longer allocates
unused screened-interaction placeholder arrays when fitting its electron basis.

`translation_gw.BlockGWResponse` retains each independent magnitude-of-transfer
Coulomb sector. Real cosine/sine partners share a block; their complex magnetic
polarization is retained. The derivative includes both delta-G W and G delta-W,
along with Hartree/Fock. DF vertices mixing transfer sectors are rejected.
This class does not retain a global internal W array for `projected_rpa` or the
older thermodynamic GW cumulant diagnostic. Use the full representation for
those separate diagnostics.

`translation_response.solve` stores Krylov unknowns in the invariant orbital
blocks. The uniform number mode remains in this space. This is a representation
of the physical translation symmetry, not a charge projection or a substituted
Ward answer. The same number/current response must agree with the full solver.

## Numerical preconditioning

The optional instantaneous-HF inverse uses the small orbital-density kernel:

\[
A_{HF}=I-BK_{HF}E,\qquad
A_{HF}^{-1}r=r+BK_{HF}(I-\chi_0K_{HF})^{-1}Er,
\quad\chi_0=EB.
\]

E extracts the endpoint density variation and B maps a static potential to
G(w+nu) delta-v G(w). All collinear spin and charge variables remain present.
This inverse is only a GMRES preconditioner. The response operator, its dynamic
functional derivative and the reported final residual remain unchanged. An
ill-conditioned HF inverse is bypassed. The application tests the HF inverse,
full-versus-blocked response and dynamic number conservation before production.

## Application checks

The physical-edge evidence includes complex block-functional contractions,
restricted-spin weighting, rejection of a momentum-mixing probe, agreement with
the full response at static and nonzero transfers, functional derivatives, and
native unmixed evaluations on the actual 32-orbital GaAs Hamiltonian. The GF2
native-functional difference is 4.2e-10. These algebraic checks do not establish
orbital, width or device convergence. Physical results must retain those gates.

The Hall application's separate charge-Newton driver solves static Hartree/Fock
at fixed dynamic self-energy, then updates the same GF2/GW functional. Final
acceptance and energies require an unmixed native GREEN map. A fixed contact
chemical potential is explicit; it must not be silently compared with a different
fixed-N or contact condition. No local GF2/GW self-energy splice is implemented.
