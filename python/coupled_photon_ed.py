"""Small fixed-N thermal reference for a magnetic DF Hamiltonian and Fock photons."""
from itertools import combinations
from math import comb, prod

import numpy as np
from scipy.linalg import eigh
from scipy.sparse import eye, kron
from scipy.sparse.linalg import eigsh
from scipy.special import logsumexp

from spatial_cavity import lift_onebody, photon_basis, validate


class CanonicalPhotonED:
    """All allowed spin sectors at fixed N; independent of the GREEN response."""

    def __init__(self, h, vertices, constant, nelectron, omega, current,
                 quadratic, cutoffs, beta, *, max_block=2200, low_states=None):
        omega, current, quadratic = validate(omega, current, quadratic)
        self.n, self.beta = current.shape[-1], float(beta)
        h, vertices = np.asarray(h), np.asarray(vertices)
        if h.shape == (self.n, self.n):
            h = np.stack((h, h))
        if (h.shape != (2, self.n, self.n) or vertices.ndim != 3
                or vertices.shape[1:] != (self.n, self.n) or not np.isfinite(self.beta) or self.beta <= 0
                or not np.isfinite(constant)
                or not isinstance(nelectron, (int, np.integer)) or not 0 <= nelectron <= 2*self.n
                or (low_states is not None and (not isinstance(low_states, int) or low_states < 2))
                or len(cutoffs) != len(omega) or not np.isfinite(h).all()
                or not np.isfinite(vertices).all()
                or np.max(abs(h-h.conj().swapaxes(-1, -2))) > 1e-10
                or np.max(abs(vertices-vertices.conj().swapaxes(-1, -2)), initial=0.) > 1e-10):
            raise ValueError('Inconsistent finite model for canonical ED')
        if any(not isinstance(n, int) or n < 2 for n in cutoffs):
            raise ValueError('Photon cutoffs must be integers of at least two')
        sectors = range(max(0, nelectron-self.n), min(nelectron, self.n)+1)
        if max(comb(self.n,up)*comb(self.n,nelectron-up) for up in sectors)*prod(cutoffs) > max_block:
            raise ValueError('ED block exceeds the supplied reference dimension bound')
        self.pdim, self.qops, q2, _, photons = photon_basis(cutoffs)
        self.blocks = []
        self.complete = True
        self.max_eigen_residual = 0.
        self.max_top_probability = 0.
        ip = eye(self.pdim)
        for up in sectors:
            down = nelectron-up
            def states(count):
                return [sum(1 << i for i in occ) for occ in combinations(range(self.n), count)]
            basis = [u | (d << self.n) for u in states(up) for d in states(down)]
            lookup = {state: i for i, state in enumerate(basis)}
            def lift(matrix):
                return lift_onebody(matrix, basis, lookup)
            ie = eye(len(basis))
            he = constant*ie + sum(lift_onebody(h[s], basis, lookup, spins=(s,)) for s in (0, 1))
            for vertex in vertices:
                op = lift(vertex)
                he += .5*(op@op-lift(vertex@vertex))
            ham = kron(he, ip, format='csr')
            for a in range(len(omega)):
                ham += kron(ie, omega[a]*photons[a], format='csr')
                ham += kron(lift(current[a]), self.qops[a], format='csr')
                for b in range(len(omega)):
                    product = q2[a] if a == b else self.qops[a]@self.qops[b]
                    ham += kron(lift(quadratic[a, b]), product, format='csr')
            defect = ham-ham.getH()
            if np.max(abs(defect.data), initial=0.) > 1e-10:
                raise ValueError('Non-Hermitian joint Hamiltonian')
            nstate = ham.shape[0]
            if low_states is not None and low_states < nstate-1:
                rng = np.random.default_rng(1967+up)
                energy, vec = eigsh(ham, k=low_states, which='SA', tol=2e-13,
                                    v0=rng.normal(size=nstate)+1j*rng.normal(size=nstate))
                order = np.argsort(energy)
                subspace, _ = np.linalg.qr(vec[:, order])
                energy, rotate = eigh(subspace.conj().T@(ham@subspace))
                vec = subspace@rotate
                self.complete = False
            else:
                energy, vec = eigh(ham.toarray(), driver='evd')
            low = vec[:, :min(4, len(energy))]
            error = np.linalg.norm(ham@low-low*energy[:low.shape[1]], axis=0).max()
            self.max_eigen_residual = max(self.max_eigen_residual, float(error))
            self.blocks.append(dict(e=energy, u=vec, basis=basis, lookup=lookup,
                                    up=up, down=down, dim=nstate))
        all_e = np.sort(np.concatenate([b['e'] for b in self.blocks]))
        self.energy = float(all_e[0])
        self.neutral_gap = float(all_e[1]-all_e[0])
        logz = logsumexp(-self.beta*(all_e-self.energy))
        self.internal_energy = 0.
        self.missing_thermal_weight_bound = 0.
        for b in self.blocks:
            p = np.exp(-self.beta*(b['e']-self.energy)-logz)
            b['p'] = p
            self.missing_thermal_weight_bound += (b['dim']-len(p))*p[-1]
            self.internal_energy += float(p@b['e'])
            thermal = (abs(b['u'])**2@p).reshape((len(b['basis']), *cutoffs)).sum(axis=0)
            b['top'] = [float(np.take(thermal, levels-1, axis=a).sum())
                        for a, levels in enumerate(cutoffs)]
        self.max_top_probability = float(np.max(np.sum([b['top'] for b in self.blocks], axis=0)))

    def density(self):
        """Spin-summed electronic density after tracing over photons."""
        rho = np.zeros((self.n, self.n), complex)
        for b in self.blocks:
            u = b['u'].reshape(len(b['basis']), self.pdim, -1)
            reduced = np.einsum('ipb,b,jpb->ij', u, b['p'], u.conj(), optimize=True)
            for i in range(self.n):
                for j in range(self.n):
                    element = np.zeros((self.n, self.n))
                    element[i, j] = 1.
                    op = lift_onebody(element, b['basis'], b['lookup'])
                    rho[j, i] += op.multiply(reduced.T).sum()
        return rho

    def spectra(self, operator, photon_source=None):
        operator = np.asarray(operator)
        source = np.zeros(len(self.qops)) if photon_source is None else np.asarray(photon_source)
        if (operator.shape != (self.n, self.n) or source.shape != (len(self.qops),)
                or np.max(abs(operator-operator.conj().T)) > 1e-10):
            raise ValueError('Expected a Hermitian common orbital probe and a photon source per mode')
        result = []
        for b in self.blocks:
            op = kron(lift_onebody(operator, b['basis'], b['lookup']), eye(self.pdim), format='csr')
            for amplitude, q in zip(source, self.qops):
                if amplitude:
                    op += amplitude*kron(eye(len(b['basis'])), q, format='csr')
            result.append(b['u'].conj().T@(op@b['u']))
        return result

    def expectation(self, matrices):
        return sum(b['p']@np.diag(a) for b, a in zip(self.blocks, matrices))

    def susceptibility(self, readout, source, nu):
        """Connected χ_AB(iν) = -δ<A>/δu_B, with thermal degeneracies at ν=0."""
        if not self.complete:
            raise ValueError('A Lehmann susceptibility requires the full intermediate-state spectrum')
        result = 0j
        for b, a, v in zip(self.blocks, readout, source):
            delta = b['e'][None, :]-b['e'][:, None]
            population = b['p'][:, None]-b['p'][None, :]
            if nu == 0:
                active = abs(delta) > 1e-10
                factor = np.broadcast_to(self.beta*b['p'][:, None], delta.shape).copy()
                np.divide(population, delta, where=active, out=factor)
            else:
                factor = population/(delta-1j*nu)
            result += np.sum(factor*a*v.T)
        if nu == 0:
            result -= self.beta*self.expectation(readout)*self.expectation(source)
        return result
