# Coupled electronic, photon and terminal response

The spatial-mode branch now computes the response of a converged GREEN GW state at finite light–matter coupling to an electronic gate, a photon force, or voltages on equilibrium leads. The native solver iterates the electron density and propagator, screened Coulomb interaction, photon displacement and covariance. The response module differentiates those same equations. All orbital matrices act in the same finite orthonormal basis. Energies and frequencies use one common unit with ℏ = 1.

## Linearized finite-coupling equations

The Hamiltonian and mean-field conventions follow the [spatial-mode implementation](spatial_cavity.md). Write the unscaled auxiliary vertices as the Coulomb factors, followed by \(J'_a=J_a+2\sum_b q_b K_{ab}\), followed by one probe per independent \(K_{ab}\). The bare auxiliary interaction has diagonal blocks \(I,-C_0,0\), with \(C_0^{-1}=\mathrm{diag}[(\nu^2+\omega_a^2)/(2\omega_a)]+2\kappa\) and \(\kappa_{ab}=\mathrm{Tr}(K_{ab}\rho)\). Thus the photon block of the joint screened interaction is \(W_{pp}=-C\). The equal-time mixed correlation is \(z=-[\Pi W]_{Kp}(0)\). The zero bare K block lets the solver measure this correlation.

At bosonic transfer \(\nu\), the subscripts L and R denote incoming equilibrium quantities at \(i\omega+i\nu\) and \(i\omega\). The linearized equations include

\[
\delta W=W_L(\delta\Pi+2P_p\delta\kappa P_p^T)W_R,\quad
\delta S=-[\delta W]_{pp}(0),\quad
\delta z=-[\delta\Pi W_R+\Pi_L\delta W]_{Kp}(0).
\]

Here \(P_p\) inserts a mode matrix into the auxiliary photon block. Both electron lines and both local vertices in \(\Pi\) vary. Each local current vertex changes by \(2K\delta q\). The dynamic self-energy has four variations, one for each vertex, one for G and one for W. The static self-energy includes the Hartree–Fock variation and \(\sum_a J'_a\delta q_a+\sum_{ab}K_{ab}\delta S_{ab}\). The oscillator equation at the *external* transfer is

\[
\delta q_a=-\sum_b [C_0(i\nu)]_{ab}
\left[f_b+\mathrm{Tr}(J'_b\delta\rho)+2\sum_c\delta z_{bc,c}\right].
\]

The solve uses \(\delta G=G_L(U+\delta\Sigma)G_R\) and this oscillator equation in one complex GMRES system. Interpolation spans both shifted legs of the intermediate-representation grid. The constructor reconstructs the native electronic self-energies, Dyson G, photon propagator, equal-time covariance, mixed correlations and coherent force before accepting a checkpoint. Each solve recomputes its own residual. A spatially constant probe at nonzero frequency tests the charge Ward identity against the actual solution.

The internal photon propagator of one-loop GW and the external response to a force are distinct observables of this approximation. The latter satisfies the tested conserved-number limit while the former retains a measurable variance error. Two-photon exchange from fluctuations of the quadratic matter vertex and higher skeleton diagrams remain outside native GW. The thermal Fock-space reference includes all terms of the specified finite Hamiltonian and provides a direct accuracy comparison. The distinction between an internal screening bubble and a source derivative follows the conserving-response construction. [Baym and Kadanoff](https://doi.org/10.1103/PhysRev.124.287)

## Electronic GF2 with dynamical photons

For a finite spatial input select `--scf_type GF2` in the native solver and
`electronic_method='GF2'` in `CoupledGWResponse`. The Hamiltonian, Gaussian
quadratic vertex, coherent field, lead embedding and full joint photon screening
are shared with GW. The choice changes the electronic Coulomb approximation.
At every iteration the dynamic electron self-energy is

\[
\Sigma_{2,C}[G]+\Sigma_{\mathrm{ring},C+\gamma}[G,q,S]
                  -\Sigma_{\mathrm{ring},C}[G].
\]

Here the ring quantities use the same dressed electron propagator at both
vertices. Native GF2 supplies the direct and exchange second-order Coulomb
diagrams. Subtracting the purely Coulomb screened interaction before the
exchange contraction leaves the mixed and photon rings. HF and the photon
tadpoles each enter once. The external derivative includes all three electron
lines of the second-Born term and the subtraction
\(W_{C,L}\,\delta\Pi_{CC}\,W_{C,R}\) for the electronic self-energy. The joint
photon propagator and mixed force use the full screening matrix.

At zero light–matter coupling this construction recovers electronic GF2. With
zero Coulomb vertices it agrees with the spatial GW solver at finite coupling.
It omits photon vertex corrections beyond the shared ring/Gaussian treatment,
including two-photon exchange from fluctuations of the quadratic vertex. Both
solvers therefore share potential photon-sector errors. Comparing them tests
the sensitivity to the electronic closure at the specified filling and fields.

## Equilibrium particle reservoirs

Optional HDF5 group `Leads` uses model `local_chain`, a zero-based orbital per terminal, and real arrays `onsite`, `hopping` and `coupling`. Each terminal is an independent semi-infinite normal chain. A chain hopping of zero gives one discrete contact orbital. The embedding accepts the finite spatial QED interface at fixed chemical potential. The open checkpoint stores `Energy_central_QED` as a central-energy diagnostic. The reservoir thermodynamic potential requires additional terms.

For contact amplitude \(v_a\), onsite \(\epsilon_a\) and hopping magnitude \(t_a\), the native and response solvers insert the surface hybridization \(\Delta_a(z)=|v_a|^2g_a(z)P_a\), with \(g_a=2/[x+\sqrt{x-2t_a}\sqrt{x+2t_a}]\), \(x=z-\epsilon_a\). Principal roots select the decaying branch in each half-plane. An explicit contact-end orbital can couple to several semiconductor orbitals through their projected tunnel amplitudes.

For a perturbation \(+u_a N_a\), integrating out the lead gives the frequency-dependent insertion \(V_a=(\Delta_{a,R}-\Delta_{a,L})/(i\nu)\). Its contribution \(\sum_a u_a V_a\) enters the *coupled* response. The particle current entering a lead is \(I_a=i[H,N_a]\). Its variation is

\[
\delta I_a=\frac{i}{\beta}\sum_{\omega,s}
\mathrm{Tr}\left[(\Delta_{a,L}-\Delta_{a,R})\delta G_s+
u_a V_a(G_{s,R}-G_{s,L})\right].
\]

The second term differentiates the reservoir line directly. A common voltage on the device and all reservoirs generates the global Ward solution. The sum of lead currents balances the charge stored in the central region at finite transfer. The measured response is \(-2\pi\delta I_a/\delta u_b\) in \(e^2/h\). At equilibrium the implementation computes positive imaginary-frequency particle admittances. A zero-temperature DC estimate requires continuous leads, a low-frequency extrapolation and independent checks on its accuracy. A discrete bath has recurrences and cannot provide that DC limit. The model supplies particle currents. Circuit displacement currents would require an electromagnetic environment. [Büttiker, Prêtre and Thomas](https://doi.org/10.1103/PhysRevLett.70.4114)

## Use and validation

`python/coupled_gw_response.py` supplies `CoupledGWResponse`. Its `solve` method returns density, displacement, covariance and mixed-response variations. Passing `terminal_voltages` also returns the two-frequency electronic variation for `LocalChainLeads.currents`. The optional lead interface is in `python/equilibrium_reservoirs.py`. The stationary photon/density checkpoint is produced by the native stable-based `mbpt.exe`.

`python/coupled_photon_ed.py` builds independent normal-ordered DF Hamiltonians in every spin sector at fixed N, with the properly projected oscillator square. Complete spectra provide connected thermal Lehmann susceptibilities including static degeneracies. A low-state calculation reports its omitted thermal population bound and requires a full intermediate spectrum before evaluating a Lehmann susceptibility.

The application repository contains the matched native/ED comparisons, external-field derivatives, neutral and charged gaps, orbital and photon cutoffs, explicit/integrated contact identity, continuum scattering reference, magnetic reversal and semiconductor calculations. [Coupled Hall benchmark](https://github.com/Chenan-Phys/green-cavity-quantum-hall/blob/main/docs/coupled_benchmark_report.md)
