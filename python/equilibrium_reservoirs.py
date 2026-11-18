"""Local noninteracting chain reservoirs and equilibrium terminal-current vertices."""
import numpy as np


class LocalChainLeads:
    """One independent semi-infinite chain per terminal, or one level when t=0."""

    def __init__(self, sites, onsite, hopping, coupling, orbitals):
        self.sites = np.asarray(sites)
        self.onsite, self.hopping, self.coupling = (np.asarray(a, dtype=float)
                                                    for a in (onsite, hopping, coupling))
        self.n = orbitals
        if (self.sites.ndim != 1 or not self.sites.size or self.sites.dtype.kind not in 'iu'
                or np.any(self.sites < 0) or np.any(self.sites >= self.n)
                or any(a.shape != self.sites.shape or not np.isfinite(a).all()
                       for a in (self.onsite, self.hopping, self.coupling))
                or np.any(self.hopping < 0)):
            raise ValueError('Invalid local-chain reservoir inputs')
        self.projectors = np.array([np.diag(np.arange(self.n) == site) for site in self.sites])

    def write(self, group):
        if len(group):
            raise ValueError('A new Leads input group is required')
        group['model'] = 'local_chain'
        group['sites'] = self.sites.astype('int32')
        for name in ('onsite', 'hopping', 'coupling'):
            group[name] = getattr(self, name)

    def hybridization(self, z):
        """One scalar retarded/advanced or Matsubara embedding per energy and lead."""
        x = np.asarray(z, complex)[:, None]-self.onsite
        if np.any(np.imag(x) == 0):
            raise ValueError('Specify the retarded or advanced half-plane explicitly')
        root = np.sqrt(x-2*self.hopping)*np.sqrt(x+2*self.hopping)
        return 2*self.coupling**2/(x+root)

    def total(self, z):
        return np.einsum('wa,aij->wij', self.hybridization(z), self.projectors)

    def voltage_vertex(self, z, nu):
        if nu <= 0:
            raise ValueError('Terminal voltage vertices use positive Matsubara transfer')
        left, right = self.hybridization(z+1j*nu), self.hybridization(z)
        return (right-left)/(1j*nu), left, right

    def currents(self, engine, response, voltages, transfer):
        """δI_a, I_a=i[H,N_a], including the direct reservoir insertion."""
        fm, _, (gr, _, _), (gl, _, _), _, nu = engine.setup(transfer)
        z = engine.mu+1j*(2*fm.n+1)*np.pi/engine.ir.beta
        vertex, left, right = self.voltage_vertex(z, nu)
        def diagonal(g):
            return np.diagonal(g, axis1=-2, axis2=-1).sum(axis=1)[:, self.sites]
        integrand = (left-right)*diagonal(response['delta_green'])
        integrand += voltages*vertex*(diagonal(gr)-diagonal(gl))
        return -1j*engine.mult*fm.backward(integrand)[-1]
