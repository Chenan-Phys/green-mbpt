"""External Matsubara response of a native spatial-cavity GW or GF2 fixed point.

Unscaled auxiliary vertices are (Coulomb, J + 2 K q, independent K probes).
The joint screened interaction has W_pp = -C. Both its frequency legs vary.
See docs/coupled_gw_response.md for conventions and approximation boundaries.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

from conserving_response import IRTransforms, TransferMaps, basis_tau, transform
from spatial_cavity import validate


def _pair(left, right, g, reverse_g, multiplicity):
    lg = np.einsum('Aij,tsjk->tsAik', left, g, optimize=True)
    rg = np.einsum('Bkl,tsli->tsBki', right, reverse_g, optimize=True)
    return -multiplicity * np.einsum('tsAik,tsBki->tAB', lg, rg, optimize=True)


def _exchange(left, right, g, w):
    lg = np.einsum('Aim,tsmn->tsAin', left, g, optimize=True)
    screened = np.einsum('tAB,tsAin->tsBin', w, lg, optimize=True)
    return -np.einsum('tsBin,Bnj->tsij', screened, right, optimize=True)


class CoupledGWResponse:
    """Differentiate the finite-Gamma, orthonormal, real-mode GREEN equations.

    electronic_method must match the native checkpoint. GF2 uses the stable
    electronic second-Born diagrams and the same photon-containing rings as GW.
    """

    def __init__(self, h, coulomb, g, sigma1, sigma_tau, mu, grid, beta,
                 omega, current, quadratic, q, covariance_q, mixed_kq, photon_w, *,
                 leads=None, electronic_method='GW'):
        import irbasis
        self.omega, self.j, self.k = validate(omega, current, quadratic)
        self.h, self.l = np.asarray(h), np.asarray(coulomb)
        self.g, self.s1, self.st = map(np.asarray, (g, sigma1, sigma_tau))
        self.mu, self.ir, self.leads = mu, IRTransforms(grid, beta), leads
        ns, self.n = self.g.shape[1:3]
        self.m = len(self.omega)
        if (ns not in (1, 2) or self.g.shape != (len(self.ir.tau), ns, self.n, self.n)
                or self.h.shape not in ((self.n, self.n), (ns, self.n, self.n))
                or self.l.ndim != 3 or self.l.shape[1:] != (self.n, self.n)
                or self.st.shape != self.g.shape or self.s1.shape != self.g.shape[1:]
                or self.j.shape[1:] != (self.n, self.n)
                or np.max(abs(self.l-self.l.conj().swapaxes(-1, -2)), initial=0.) > 1e-10):
            raise ValueError('Inconsistent finite-system checkpoint or Coulomb vertices')
        self.mult = 2 if ns == 1 else 1
        self.q, self.sq = np.asarray(q), np.asarray(covariance_q)
        self.pairs = [(a, b) for a in range(self.m) for b in range(a, self.m)]
        nc = len(self.l)
        if electronic_method not in ('GW', 'GF2'):
            raise ValueError('Expected GW or GF2 electron correlation')
        self.second_born = electronic_method == 'GF2'
        self.cc = slice(0,nc)
        if self.second_born:
            from gf2_response import coulomb_tensor
            # The explicit lead ends have exactly zero Coulomb matrix elements.
            self.active = np.flatnonzero(np.any(self.l != 0, axis=(0,1)))
            self.active_pair = np.ix_(self.active,self.active)
            self.eri = coulomb_tensor(self.l[:,self.active_pair[0],self.active_pair[1]])
        self.ph = slice(nc, nc+self.m)
        self.probes = slice(nc+self.m, None)
        if (self.q.shape != (self.m,) or self.sq.shape != (self.m, self.m)
                or np.shape(mixed_kq) != (len(self.pairs), self.m)
                or np.shape(photon_w) != (len(self.ir.bn), self.m, self.m)):
            raise ValueError('Inconsistent photon checkpoint')
        self.je = self.j + 2*np.einsum('b,abij->aij', self.q, self.k)
        self.v = np.concatenate((self.l, self.je, [self.k[a, b] for a, b in self.pairs]))
        self.instant = np.zeros((len(self.v), len(self.v)))
        self.instant[:nc, :nc] = np.eye(nc)
        self.rho = -self.mult*self.g[-1].sum(axis=0)
        self.kappa = self._trace(self.k, self.rho)
        self.pi = self.ir.b_to_w(_pair(self.v, self.v, self.g, self.g[::-1], self.mult))
        self.w = self._screen(self.ir.bomega, self.pi)
        self.sw = self.ir.f_to_w(self.st)
        wc = self.w-self.instant
        if self.second_born:
            wc[:, self.cc, self.cc] = (self.w[:, self.cc, self.cc]
                                        - self._electronic_screen(self.pi))
        sigma_t = _exchange(self.v, self.v, self.g, self.ir.w_to_btau(wc))
        if self.second_born:
            sigma_t += self._second_order(self.g,self.g,self.g[::-1])
        static = self._hf(-self.g[-1]) + np.einsum('a,aij->ij', self.q, self.j)
        static += np.einsum('ab,abij->ij', np.outer(self.q, self.q)+self.sq, self.k)
        z = -self.ir.w_to_btau((self.pi@self.w)[:, self.probes, self.ph])[0]
        c = -self.w[:, self.ph, self.ph]
        # S uses exactly the native analytic oscillator baseline and the IR
        # transform of the decaying correction, including off-diagonal modes.
        r = np.diag(np.sqrt(self.omega))
        stiffness = np.diag(self.omega)+4*self.kappa.real
        e, u = np.linalg.eigh(r@stiffness@r)
        if e.min() <= 0:
            raise ValueError('Unstable quadratic photon stiffness')
        freq = np.sqrt(e)
        sq0 = r@(u*(1/(np.tanh(.5*beta*freq)*freq)))@u.T@r
        self.fixed_point = {
            'green': np.max(abs(self.ir.w_to_f(self._dyson(self.ir.fomega, self.sw))-self.g)),
            'sigma_static': np.max(abs(static-self.s1)),
            'sigma_dynamic': np.max(abs(sigma_t-self.st)),
            'photon_propagator': np.max(abs(c-photon_w)),
            'covariance_q': np.max(abs(sq0+self.ir.w_to_btau(c-self._c0(self.ir.bomega))[0]-self.sq)),
            'mixed_kq': np.max(abs(z-mixed_kq)),
            'mean_force': np.max(abs(.5*stiffness@self.q+self._trace(self.j, self.rho)+2*self._mixed_force(z))),
        }
        if not all(np.isfinite(x) and x < 1e-7 for x in self.fixed_point.values()):
            raise ValueError(f'Coupled checkpoint fails independent reconstruction: {self.fixed_point}')
        self.fb, self.bb = (irbasis.load(kind, self.ir.lambda_) for kind in ('F', 'B'))
        fit = np.linalg.pinv(basis_tau(self.fb, self.ir.fx, beta))
        self.gc = transform(fit, self.g[1:-1])
        self.sc = transform(fit, self.st[1:-1])
        bfit = np.linalg.pinv(self.bb.compute_unl(self.ir.bn)*np.sqrt(beta))
        self.pic = transform(bfit, self.pi)
        self.wc = transform(bfit, wc)
        quadrature = np.polynomial.legendre.leggauss(256)[0]
        self.x = np.unique(np.r_[-1., self.ir.fx, self.ir.bx, quadrature, 1.])
        self.gt = transform(basis_tau(self.fb, self.x, beta), self.gc)
        self.wt = transform(basis_tau(self.bb, self.x, beta), self.wc)
        self.tangent = None
        self._setups = {}

    @staticmethod
    def _trace(operators, rho):
        return np.einsum('...ij,ji->...', operators, rho)

    def _hf(self, rho):
        charge = self._trace(self.l, self.mult*rho.sum(axis=0))
        return (np.einsum('a,aij->ij', charge, self.l)
                - np.einsum('aim,smn,anj->sij', self.l, rho, self.l, optimize=True))

    def _second_order(self, a, b, c):
        from gf2_response import cubic
        out = np.zeros_like(a)
        if self.active.size:
            i,j = self.active_pair
            out[...,i,j] = cubic(a[...,i,j],b[...,i,j],c[...,i,j],self.eri,self.mult)
        return out

    def _electronic_screen(self, pi):
        return np.linalg.inv(np.eye(len(self.l))-pi[:,self.cc,self.cc])

    def _c0(self, nu):
        diagonal = (np.asarray(nu)[:, None]**2+self.omega**2)/(2*self.omega)
        return np.linalg.inv(2*self.kappa+diagonal[:, :, None]*np.eye(self.m))

    def _screen(self, nu, pi):
        bare = np.broadcast_to(self.instant, pi.shape).astype(complex)
        bare[:, self.ph, self.ph] = -self._c0(nu)
        return np.linalg.solve(np.eye(len(self.v))-bare@pi, bare)

    def _dyson(self, freq, sigma_w):
        z = 1j*freq+self.mu
        inverse = z[:, None, None, None]*np.eye(self.n)-self.h-self.s1-sigma_w
        if self.leads is not None:
            inverse -= self.leads.total(z)[:, None]
        return np.linalg.inv(inverse)

    def _mixed_force(self, mixed):
        force = np.zeros(self.m, dtype=complex)
        for index, (a, b) in enumerate(self.pairs):
            force[a] += mixed[index, b]
            if a != b:
                force[b] += mixed[index, a]
        return force

    def setup(self, transfer):
        if not isinstance(transfer, (int, np.integer)) or transfer < 0:
            raise ValueError('Nonnegative integer transfer required')
        if transfer in self._setups:
            return self._setups[transfer]
        # Multiple probes normally share one transfer. Limit memory when
        # scanning a spectrum with many different two-leg IR grids.
        if len(self._setups) >= 2:
            self._setups.pop(next(iter(self._setups)))
        beta = self.ir.beta
        fm = TransferMaps(self.fb, self.ir.fn, self.x, beta, transfer)
        bm = TransferMaps(self.bb, self.ir.bn, self.x, beta, transfer)
        nu = 2*np.pi*transfer/beta
        def leg(shift):
            fermions, bosons = fm.n+shift, bm.n+shift
            st = transform(self.fb.compute_unl(fermions)*np.sqrt(beta), self.sc)
            pi = transform(self.bb.compute_unl(bosons)*np.sqrt(beta), self.pic)
            g = self._dyson((2*fermions+1)*np.pi/beta, st)
            return g, pi, self._screen(2*np.pi*bosons/beta, pi)
        right, left = leg(0), leg(transfer)
        phase = np.exp(1j*np.pi*transfer*(self.x+1))
        self._setups[transfer] = (fm, bm, right, left, phase, nu)
        return self._setups[transfer]

    def _variation(self, dg_w, dq, setup):
        fm, bm, (_, _, wr), (_, pil, wl), phase, nu = setup
        dg = fm.backward(dg_w)
        drho = -dg[-1]
        total = self.mult*drho.sum(axis=0)
        dk = self._trace(self.k, total)
        dv = np.zeros_like(self.v)
        dv[self.ph] = 2*np.einsum('b,abij->aij', dq, self.k)
        # A varying vertex at t2 carries exp(+i nu (t1-t2)) in this convention.
        dp = _pair(self.v, self.v, dg, self.gt[::-1], self.mult)
        dp += _pair(dv, self.v, self.gt, self.gt[::-1], self.mult)
        dp += phase[:, None, None]*(
            _pair(self.v, self.v, self.gt, dg[::-1], self.mult)
            + _pair(self.v, dv, self.gt, self.gt[::-1], self.mult))
        dpi = bm.forward(dp)
        kernel = dpi.copy()
        kernel[:, self.ph, self.ph] += 2*dk
        dw = wl@kernel@wr
        dwt = bm.backward(dw)
        ds = -dwt[0, self.ph, self.ph]
        dz = -bm.backward((dpi@wr+pil@dw)[:, self.probes, self.ph])[0]
        if self.second_born:
            # Only the electron exchange subtracts the pure Coulomb ring.
            # ds and dz above use the full joint photon solution.
            cc = self.cc
            dw[:,cc,cc] -= self._electronic_screen(pil)@dpi[:,cc,cc]@self._electronic_screen(setup[2][1])
            dwt = bm.backward(dw)
        static = self._hf(drho) + np.einsum('a,aij->ij', dq, self.je)
        static += np.einsum('ab,abij->ij', ds, self.k)
        dynamic = _exchange(self.v, self.v, dg, self.wt)
        dynamic += _exchange(self.v, self.v, self.gt, dwt)
        dynamic += _exchange(dv, self.v, self.gt, self.wt)
        dynamic += phase[:, None, None, None]*_exchange(self.v, dv, self.gt, self.wt)
        if self.second_born:
            i,j = self.active_pair
            cache_bytes = 2*len(self.x)*dg.shape[1]**2*len(i)**4*16
            if len(i) and cache_bytes <= 384*1024**2:
                if self.tangent is None:
                    from second_born_tangent import SecondBornTangent
                    self.tangent = SecondBornTangent(self.gt[...,i,j],self.eri,self.mult)
                dynamic[...,i,j] += self.tangent.apply(dg[...,i,j],phase)
            else:
                dynamic += self._second_order(dg,self.gt,self.gt[::-1])
                dynamic += self._second_order(self.gt,dg,self.gt[::-1])
                dynamic += phase[:,None,None,None]*self._second_order(self.gt,self.gt,dg[::-1])
        force = self._trace(self.je, total) + 2*self._mixed_force(dz)
        return static+fm.forward(dynamic), force, drho, ds, dz

    def solve(self, operator, transfer, photon_source=None, rtol=2e-11, maxiter=30,
              *, terminal_voltages=None):
        """Response to +u exp(-i nu tau) [dGamma(operator) + photon_source.Q].

        Returned arrays are variations, not positive-sign susceptibilities.
        An operator may be shared across spins or specified per stored spin.
        """
        operator = np.asarray(operator)
        if (operator.shape not in ((self.n, self.n), self.s1.shape)
                or not np.isfinite(operator).all()
                or np.max(abs(operator-operator.conj().swapaxes(-1, -2))) > 1e-10):
            raise ValueError('A finite Hermitian electronic source is required')
        source = np.zeros(self.m) if photon_source is None else np.asarray(photon_source)
        if source.shape != (self.m,) or not np.isfinite(source).all() or np.any(np.imag(source)):
            raise ValueError('Supply one real source per standing photon mode')
        setup = self.setup(transfer)
        _, _, (gr, _, _), (gl, _, _), _, nu = setup
        shape, size = gr.shape, gr.size
        c0 = self._c0([nu])[0]
        field = operator
        if terminal_voltages is not None:
            voltages = np.asarray(terminal_voltages)
            if (self.leads is None or voltages.shape != self.leads.sites.shape
                    or not np.isfinite(voltages).all() or np.any(np.imag(voltages))):
                raise ValueError('A real voltage per attached terminal is required')
            z = self.mu+1j*(2*setup[0].n+1)*np.pi/self.ir.beta
            vertex, _, _ = self.leads.voltage_vertex(z, nu)
            field = operator+np.einsum('a,wa,aij->wij', voltages, vertex, self.leads.projectors)[:, None]
        rhs = np.r_[(gl@field@gr).ravel(), -c0@source]

        def action(x):
            dg, dq = x[:size].reshape(shape), x[size:]
            dsigma, force, *_ = self._variation(dg, dq, setup)
            return np.r_[(dg-gl@dsigma@gr).ravel(), dq+c0@force]

        trace = []
        linear = LinearOperator((len(rhs), len(rhs)), matvec=action, dtype=complex)
        x, info = gmres(linear, rhs, x0=rhs.copy(), rtol=rtol, atol=1e-14,
                        maxiter=maxiter, restart=30, callback_type='pr_norm',
                        callback=lambda value: trace.append(float(value)))
        residual = float(np.linalg.norm(action(x)-rhs)/max(np.linalg.norm(rhs), 1e-30))
        if info or not np.isfinite(residual) or residual > max(10*rtol, 2e-9):
            raise RuntimeError(f'Coupled response did not converge: info={info}, residual={residual}')
        dg, dq = x[:size].reshape(shape), x[size:]
        _, _, drho, ds, dz = self._variation(dg, dq, setup)
        result = dict(drho=drho, dq=dq, d_covariance_q=ds, d_mixed_kq=dz,
                      residual=residual, iterations=len(trace), nu=nu)
        if terminal_voltages is not None:
            result['delta_green'] = dg
        global_drive = self.leads is None or (terminal_voltages is not None and np.all(voltages == 1))
        if transfer and global_drive and np.max(abs(operator-np.eye(self.n))) < 1e-13 and not source.any():
            ward = (gr-gl)/(1j*nu)
            result['ward_solution_error'] = float(np.linalg.norm(dg-ward)/np.linalg.norm(ward))
        return result
