"""Schema-2 spatial photons and independent finite-Fock-space references.

Atomic units (or one consistently chosen model energy unit) throughout.
H = He + sum omega a†a + sum Q J + sum_(mu,nu) Q_mu Q_nu K_mu,nu.
Q=a+a†; all sums over quadratic mode indices are ORDERED. Current/quadratic
operators include zero-point amplitudes. This module does not solve Maxwell's
equations, infer a device dielectric function, or silently choose a gauge.
"""
import itertools
import numpy as np


def validate(omega, current, quadratic):
    omega = np.asarray(omega, dtype=float)
    current = np.asarray(current, dtype=complex)
    quadratic = np.asarray(quadratic, dtype=complex)
    if omega.ndim != 1 or not 1 <= len(omega) <= 4 or not np.isfinite(omega).all() or min(omega) <= 0:
        raise ValueError('Provide one to four positive finite bare photon energies')
    m = len(omega)
    if current.ndim != 3 or current.shape[0] != m or current.shape[1] != current.shape[2]:
        raise ValueError('current must have shape (mode,orbital,orbital)')
    n = current.shape[1]
    if quadratic.shape != (m, m, n, n):
        raise ValueError('quadratic must have shape (mode,mode,orbital,orbital)')
    for name, x in [('current', current), ('quadratic', quadratic)]:
        if not np.isfinite(x).all() or np.max(abs(x-x.swapaxes(-1,-2).conj())) > 1e-11:
            raise ValueError(name+' must be finite and Hermitian in orbital indices')
    if np.max(abs(quadratic-quadratic.swapaxes(0,1))) > 1e-11:
        raise ValueError('quadratic must be symmetric in mode indices')
    return omega, current, quadratic


def write(group, omega, current, quadratic):
    """Write into a NEW h5py group; preserve complex matrices without casting."""
    omega, current, quadratic = validate(omega, current, quadratic)
    if len(group):
        raise ValueError('Refusing to overwrite an existing QED group')
    group['schema'] = 2
    group['model'] = 'current_quadratic'
    group['finite_system'] = 1
    group['omega'] = omega
    for name, x in [('current', current), ('quadratic', quadratic)]:
        group[name] = np.stack((x.real, x.imag), axis=-1)


def photon_basis(cutoffs):
    """Projected Q, Q², P², a†a. Q_N@Q_N is NOT projected Q² at its top level."""
    from scipy.sparse import csr_matrix, diags, eye, kron
    if any(not isinstance(n, int) or n < 2 for n in cutoffs):
        raise ValueError('Each photon cutoff must be an integer >=2')
    dim = int(np.prod(cutoffs))

    def embed(a, mode):
        result = csr_matrix([[1.]])
        for index, n in enumerate(cutoffs):
            result = kron(result, a if index == mode else eye(n), format='csr')
        return result

    qq, q2, p2, nn = [], [], [], []
    for mode, n in enumerate(cutoffs):
        a = diags(np.sqrt(np.arange(1, n)), 1, shape=(n,n), format='csr')
        q = a+a.T
        # Projection of the full oscillator operator, including the top-state
        # contact term that is lost by multiplying truncated Q matrices.
        top = csr_matrix(([float(n)], ([n-1], [n-1])), shape=(n,n))
        qq.append(embed(q, mode))
        q2.append(embed(q@q+top, mode))
        p = 1j*(a.T-a)
        p2.append(embed(p@p+top, mode))
        nn.append(embed(diags(np.arange(n),dtype=float), mode))
    return dim, qq, q2, p2, nn


def projected_phase(eta, cutoff):
    """Exact Fock matrix <n|exp(i eta Q)|m>, projected AFTER exponentiation."""
    from scipy.special import eval_genlaguerre, gammaln
    out = np.empty((cutoff,cutoff), complex)
    for n in range(cutoff):
        for m in range(cutoff):
            lo, hi = min(n,m), max(n,m)
            out[n,m] = (np.exp(-eta*eta/2+(gammaln(lo+1)-gammaln(hi+1))/2)*
                (1j*eta)**(hi-lo)*eval_genlaguerre(lo,hi-lo,eta*eta))
    return out


def electronic_basis(norb, nelectron):
    nup, ndn = (nelectron+1)//2, nelectron//2
    up = [sum(1<<i for i in c) for c in itertools.combinations(range(norb), nup)]
    dn = [sum(1<<i for i in c) for c in itertools.combinations(range(norb), ndn)]
    basis = [a | b<<norb for a in up for b in dn]
    return basis, {s:i for i,s in enumerate(basis)}


def lift_onebody(matrix, basis, lookup, *, spins=(0, 1)):
    """Bit-string second quantization in the specified fixed-spin sector."""
    from scipy.sparse import coo_matrix
    n = len(matrix)
    rows, cols, values = [], [], []
    for col, state in enumerate(basis):
        for spin in spins:
            for j in range(n):
                bj = spin*n+j
                if not (state>>bj)&1:
                    continue
                removed = state^(1<<bj)
                sign_j = -1 if (state&((1<<bj)-1)).bit_count()%2 else 1
                for i in range(n):
                    bi = spin*n+i
                    if (removed>>bi)&1 or abs(matrix[i,j])<1e-16:
                        continue
                    sign_i = -1 if (removed&((1<<bi)-1)).bit_count()%2 else 1
                    rows.append(lookup[removed | (1<<bi)])
                    cols.append(col)
                    values.append(sign_i*sign_j*matrix[i,j])
    return coo_matrix((values,(rows,cols)),shape=(len(basis),)*2).tocsr()


def exact_density_model(h, pair, constant, nelectron, omega, current, quadratic, cutoffs,
                        full_peierls=None):
    """Independent fixed-N ED for a density interaction plus spatial photons.

    Optional full_peierls=(hopping,eta[mode,i,j]) replaces hopping by the exact
    projected exponential; it is a DIFFERENT Hamiltonian from the quadratic one.
    Never compare their energies as if only a solver tolerance separated them.
    """
    from scipy.sparse import diags, eye, kron
    from scipy.sparse.linalg import eigsh
    omega, current, quadratic = validate(omega,current,quadratic)
    n, m = len(h), len(omega)
    if len(cutoffs) != m:
        raise ValueError('One cutoff per mode is required')
    basis, lookup = electronic_basis(n,nelectron)
    occ = np.array([[((s>>i)&1)+((s>>(n+i))&1) for i in range(n)] for s in basis])
    interaction = .5*(np.einsum('bi,ij,bj->b',occ,pair,occ)-occ@np.diag(pair))+constant
    he = lift_onebody(h,basis,lookup)+diags(interaction)
    pdim, q, q2, p2, numbers = photon_basis(cutoffs)
    edim = len(basis)
    ham = kron(he,eye(pdim),format='csr')
    for a in range(m):
        ham += kron(eye(edim),omega[a]*numbers[a],format='csr')
    if full_peierls is None:
        for a in range(m):
            ham += kron(lift_onebody(current[a],basis,lookup),q[a],format='csr')
            for b in range(m):
                qp = q2[a] if a==b else q[a]@q[b]
                ham += kron(lift_onebody(quadratic[a,b],basis,lookup),qp,format='csr')
    else:
        hopping, eta = full_peierls
        for i in range(n):
            for j in range(n):
                if i==j or abs(hopping[i,j])<1e-16:
                    continue
                hop = np.zeros_like(h)
                hop[i,j] = hopping[i,j]
                phase = np.ones((1,1),complex)
                for a, cutoff in enumerate(cutoffs):
                    phase = np.kron(phase,projected_phase(eta[a,i,j],cutoff))
                ham += kron(lift_onebody(hop,basis,lookup),phase-np.eye(pdim),format='csr')
    herm = ham-ham.getH()
    if herm.nnz and np.max(abs(herm.data))>1e-11:
        raise RuntimeError('Non-Hermitian exact Hamiltonian')
    rng = np.random.default_rng(185)
    vals, vecs = eigsh(ham,k=2,which='SA',tol=2e-13,
                      v0=rng.normal(size=ham.shape[0])+1j*rng.normal(size=ham.shape[0]))
    order = np.argsort(vals)
    vals, psi = vals[order], vecs[:,order[0]]
    amplitudes = psi.reshape(edim,pdim)
    eprob = np.sum(abs(amplitudes)**2,axis=1)

    def expected_ph(operator):
        return float(sum(np.vdot(row,operator@row).real for row in amplitudes))

    qmean = np.array([expected_ph(o) for o in q])
    sq = np.array([[expected_ph(q2[a] if a==b else q[a]@q[b])-
                    qmean[a]*qmean[b] for b in range(m)] for a in range(m)])
    spdiag = np.array([expected_ph(o) for o in p2])
    phprob = np.sum(abs(amplitudes)**2,axis=0).reshape(cutoffs)
    top_weight = [float(np.take(phprob,cutoffs[a]-1,axis=a).sum()) for a in range(m)]
    return dict(energy=float(vals[0]), gap=float(vals[1]-vals[0]), dimension=ham.shape[0],
        electronic_dimension=edim, photon_cutoffs=cutoffs,
        eigen_residual=float(np.linalg.norm(ham@psi-vals[0]*psi)),
        site_density=(eprob@occ).tolist(), q_mean=qmean.tolist(), covariance_Q=sq.tolist(),
        P2_diagonal=spdiag.tolist(), photon_number=[expected_ph(o) for o in numbers],
        photon_top_weight=top_weight)
