# Spatial current and quadratic photon modes

Implementation branch: `cavity-spatial-modes`, based directly on stable GREEN
v1.0.0 (`d593a9c`) plus the finite-magnetic-response correction (`25c893a`).
The older `cavity-general` molecular dipole branch is separate. This interface
does not inherit or claim to repair its unresolved origin/number-response test.

## Hamiltonian and scope

The explicit finite-basis model is

\[
H=H_e+\sum_\mu\omega_\mu a^\dagger_\mu a_\mu
 +\sum_\mu Q_\mu\hat J_\mu
 +\sum_{\mu\nu}Q_\mu Q_\nu\hat K_{\mu\nu},\qquad
Q_\mu=a_\mu+a^\dagger_\mu.
\]

The last sum contains **both ordered pairs** when mu differs from nu. J and K
are Hermitian one-electron matrices; K is symmetric in its two mode indices.
They already contain the zero-point field amplitude. Q and P=i(a†−a) are
dimensionless, J/K/omega have units of energy, and the free photon zero-point
constant is omitted. Photon fluctuations and their frequency dependence remain
explicit; this is not a fitted static attractive potential.

For continuum minimal coupling, J is the symmetrized current coupling to the
spatial mode profile and K is the independently projected diamagnetic A²
operator. K cannot in general be reconstructed by squaring a projected J.
For a tight-binding model, expand the SAME Peierls exponential on every bond:

\[
h_{ij}(Q)=h^0_{ij}e^{i\sum_\mu\eta^\mu_{ij}Q_\mu},\quad
J^\mu_{ij}=i\eta^\mu_{ij}h^0_{ij},\quad
K^{\mu\nu}_{ij}=-\tfrac12\eta^\mu_{ij}\eta^\nu_{ij}h^0_{ij}.
\]

The implemented Hamiltonian is the **quadratic expansion**, not the full bond
exponential. Higher Peierls terms change energies at order eta^4 in the weak
field limit. An independent projected full-exponential ED reference is supplied
to measure that difference. A transverse spatial field profile, its normalization
and its compatibility with electrostatic boundaries are input physics, not
provided by this module; there is no Maxwell/device-mode solver here.

Initial supported domain: one to four real oscillator coordinates, finite
Gamma-point orthonormal orbitals, scalar one/two-spin HF or GW, CPU double
precision, identity spatial symmetry, no antiunitary reduction, one MPI rank.
Unsupported settings fail explicitly. Without a QED group the existing
electronic path is unchanged. The process-local photon state is reset by each
Dyson construction; concurrent independent Dyson instances in one process are
not supported.

Both rotating and counter-rotating linear terms are retained through a+a†.
The input describes real standing-wave vector-potential coordinates. A complex
traveling/circular mode needing independent Q and P couplings is not included
in this first interface. Mean-field positive stiffness is checked at runtime;
for a Peierls expansion, global stability and omitted higher orders still need
an explicit small-coupling/cutoff check.

## Approximation

HF uses a self-consistent product of an electronic HF state and a displaced,
squeezed Gaussian photon state. GW additionally iterates the full electronic
G, joint Coulomb/current screening, the photon two-point function, the density,
and the coherent photon mean. Quadratic coupling contributes a density-dependent
photon stiffness and a photon-covariance electronic tadpole. The mixed K/current
response is retained in the coherent mean-force equation.

Two-photon exchange from fluctuations of K times fluctuations of Q², and vertex
corrections beyond the current bubble, are **not included**. Thus this is a
one-loop current-vertex GW approximation with quadratic tadpoles, not an exact
solution of the quadratic Hamiltonian. Weak-coupling and exact limiting checks
are necessary. Internal bare-vertex GW response need not obey every exact Ward
identity, even when the electronic fixed point converges. Orbital-basis
covariance is not a proof of equivalence between electromagnetic gauges.

## Equations and signs

Let rho be the total positive electronic density, kappa_mu,nu=Tr(K_mu,nu rho),
j_mu=Tr(J_mu rho), q=<Q>, S=<delta Q delta Q>, and
z_mu=sum_nu <delta K_mu,nu delta Q_nu>. Then

\[
[\operatorname{diag}\omega+4\kappa]q=-2(j+2z),\quad
J'_\mu=J_\mu+2\sum_\nu q_\nu K_{\mu\nu},
\]
\[
\Sigma_\mathrm{ph}^{\rm static}
=\sum_\mu q_\mu J_\mu
 +\sum_{\mu\nu}(q_\mu q_\nu+S_{\mu\nu})K_{\mu\nu},
\quad
C_0(i\nu)=\left[\operatorname{diag}\frac{\nu^2+\omega^2}{2\omega}
 +2\kappa\right]^{-1}.
\]

The photon covariance C is defined as positive <T delta Q delta Q>. Native
electronic polarization Pi uses the opposite sign to a density covariance.
Scale the photon vertices as L_mu=sqrt(2/omega_mu) J'_mu and let
s=diag(sqrt(omega/2)). In the augmented auxiliary space,

\[
B=\operatorname{blockdiag}(I_\mathrm{Coulomb},-sC_0s,0_\mathrm{probe}),\quad
W=(I-B\Pi)^{-1}B,\quad
W_c=W-\operatorname{blockdiag}(I_\mathrm{Coulomb},0,0).
\]

The attractive retarded photon block is negative. A positive square root of B
would implement the wrong kernel. Only the instantaneous Coulomb block is
subtracted: the bare photon exchange remains in Sigma(tau)=−L G L Wc(tau).
The full complex LU solve retains both halves of the magnetic tau response.

With chi=Pi+Pi W Pi and chi_J=s chi_ph,ph s,

\[
C=C_0-C_0\chi_J C_0,\quad
\delta S=\beta^{-1}\sum_\nu(C-C_0),\quad
\delta\langle PP\rangle=-\beta^{-1}\sum_\nu
 \nu^2\omega^{-1}(C-C_0)\omega^{-1}.
\]

The bare, squeezed finite-temperature Q/P covariances are evaluated analytically
from the positive eigenvalues of
sqrt(omega)[diag(omega)+4 kappa]sqrt(omega). IR transforms sum only the decaying
corrections, avoiding a numerical contact term. Instability or nonpositive
covariances trigger errors; no negative eigenvalue is silently clipped.

Additional K vertices have zero bare interaction and act only as response
probes. Their mixed susceptibility with J' supplies z; they introduce no
two-photon self-energy. The connected mixed linear energy is
X=beta^−1 sum_nu Tr(C0 chi_J). Electron/photon state and density residuals enter
the stopping condition in addition to native electronic energy convergence.
The full imaginary-time electronic propagator must also stop changing: density
and energy alone can be insensitive to a still-changing dynamical self-energy.

## Energy accounting

Native `Energy_HF` and `Energy_2b` retain their original electronic Galitskii–
Migdal meaning; their sum is **not** the full electron–photon energy. The added
dataset is

\[
E_\mathrm{total,QED}=E_\mathrm{GM,e}
 +\tfrac12\operatorname{Tr}(\rho\Sigma_\mathrm{ph}^{\rm static})
 +\tfrac12 X+E_\mathrm{ph},\quad
E_\mathrm{ph}=\sum_\mu\frac{\omega_\mu}{4}
 [q_\mu^2+S_{\mu\mu}+\langle P_\mu^2\rangle-2].
\]

The correction restores the other half of the photon coupling already counted
in the electronic GM expression and adds the oscillator energy. The checkpoint
stores `Energy_total_QED`, every correction, coherent means, Q/P covariances,
mixed KQ response and sampled C(i nu). These energies are expectation energies;
they are not automatically finite-temperature free energies.

## Input and reference utilities

`python/spatial_cavity.py` validates and writes a new HDF5 `QED` group:
`schema=2`, `model=current_quadratic`, `finite_system=1`, `omega[M]`,
`current[M,n,n,2]`, `quadratic[M,M,n,n,2]`. The final dimension stores real and
imaginary parts. Existing electronic input/DF files keep the stable v1.0.0
schema; photon/probe vertices are appended in memory, not to Coulomb DF files.

That module also contains independent fixed-particle-number electron–photon ED
references. They project Q² from the full oscillator algebra (including the
top-level contact term), and use analytic projected displacement matrix
elements for the full-Peierls comparison. Photon cutoff and eigenvector
residual checks are required before using an ED value as a reference.

Hall geometry, line integration of illustrative mode profiles, run scripts,
saved numerical acceptance checks and figures belong to the separate
[cavity-Hall project](https://github.com/Chenan-Phys/green-cavity-quantum-hall).
The mode tests do not establish a reconstructed GaAs edge or Hall resistance.
