"""Matrix-free linear response of finite-system self-consistent electronic GW.

Research reference, not a production C++ response kernel. Requires a converged
stable-GREEN finite orthonormal Gamma-point checkpoint with Hermitian Coulomb
DF vertices. A bosonic transfer nu uses delta G(w+nu,w) and the convention
delta G(t1,t2)=exp(-i nu t1) delta g_nu(t1-t2). Both delta G W and G delta W
are differentiated, together with Hartree and exchange. No density projection
or Ward-identity substitution is used in the solve. See docs/conserving_response.md.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres


def transform(matrix, array):
    return (matrix@array.reshape(array.shape[0],-1)).reshape((len(matrix),)+array.shape[1:])


class IRTransforms:
    def __init__(self, grid, beta):
        import h5py
        with h5py.File(grid) as f:
            def rep(kind):
                ux,uw=np.asarray(f[kind+'/uxl']),np.asarray(f[kind+'/uwl'])
                tc=np.vstack((f[kind+'/u1l_neg'],ux,f[kind+'/u1l_pos']))*np.sqrt(2/beta)
                ct=np.linalg.pinv(ux)*np.sqrt(beta/2)
                nc,cn=uw*np.sqrt(beta),np.linalg.pinv(uw)/np.sqrt(beta)
                return tc,ct,nc,cn
            ft,fc,fn,fni=rep('fermi')
            bt,bc,bn,bni=rep('bose')
            f_to_b=bt@bc@np.asarray(f['fermi/uxl_other'])*np.sqrt(2/beta)
            b_to_f=ft@fc@np.asarray(f['bose/uxl_other'])*np.sqrt(2/beta)
            self.ff=fn@fc
            self.tf=ft@fni
            self.fb=bn@bc@f_to_b[1:-1]@fc
            self.tb=b_to_f@bni
            self.fomega=(2*np.asarray(f['fermi/ngrid'])+1)*np.pi/beta
            self.bomega=2*np.pi*np.asarray(f['bose/ngrid'])/beta
            self.fn=np.asarray(f['fermi/ngrid'])
            self.bn=np.asarray(f['bose/ngrid'])
            self.fx=np.asarray(f['fermi/xgrid'])
            self.bx=np.asarray(f['bose/xgrid'])
            self.lambda_=float(f['fermi/metadata/lambda'][()])
            self.fc,self.bni=fc,bni
            self.tau=(np.r_[-1.,np.asarray(f['fermi/xgrid']),1.]+1)*beta/2
        self.beta=beta

    def f_to_w(self, a):
        return transform(self.ff,a[1:-1])

    def b_to_w(self, a):
        return transform(self.fb,a[1:-1])

    def w_to_f(self, a):
        return transform(self.tf,a)

    def w_to_btau(self, a):
        return transform(self.tb,a)


def basis_tau(basis,x,beta):
    # The installed irbasis version's vector-index path has a NumPy broadcasting
    # defect. Scalar-index evaluation is the supported equivalent operation.
    return np.stack([basis.ulx(i,x) for i in range(basis.dim())],axis=1)*np.sqrt(2/beta)


class TransferMaps:
    """Two-leg IR span U_l(w) plus U_l(w+nu), with weighted orthogonalization.

    A single equilibrium IR span cannot resolve both sets of shifted poles.
    Time orthogonalization removes near-null overcomplete directions BEFORE
    forming the frequency inverse, avoiding cancellation of huge coefficients.
    """
    def __init__(self,basis,base_n,x,beta,m,cutoff=1e-10):
        # Oversample both low-frequency legs. The union of sparse equilibrium
        # nodes alone can leave nearly invisible interpolation directions at
        # intermediate transfers, even after time-space orthogonalization.
        low=np.arange(-32,33)
        self.n=np.unique(np.r_[base_n,base_n-m,low,low-m])
        all_n=np.unique(np.r_[self.n,self.n+m])
        raw=basis.compute_unl(all_n)*np.sqrt(beta)
        tw=raw[np.searchsorted(all_n,self.n)]
        tt=basis_tau(basis,x,beta)
        if m:
            tw=np.c_[tw,raw[np.searchsorted(all_n,self.n+m)]]
            tt=np.c_[tt,np.exp(1j*np.pi*m*(x+1))[:,None]*tt]
        weight_t=np.sqrt(np.r_[x[1]-x[0],(x[2:]-x[:-2])/2,x[-1]-x[-2]])
        weight_w=np.sqrt(np.maximum(np.r_[self.n[1]-self.n[0],
            (self.n[2:]-self.n[:-2])/2,self.n[-1]-self.n[-2]],1.))
        u,s,vh=np.linalg.svd(weight_t[:,None]*tt,full_matrices=False)
        keep=s>s[0]*cutoff
        qt=u[:,keep]/weight_t[:,None]
        qw=tw@vh[keep].conj().T/s[keep]
        self.to_time=qt@np.linalg.pinv(weight_w[:,None]*qw,rcond=1e-10)*weight_w[None,:]
        self.to_frequency=qw@(u[:,keep].conj().T*weight_t[None,:])
        self.rank=int(sum(keep))

    def forward(self,a):
        return transform(self.to_frequency,a)

    def backward(self,a):
        return transform(self.to_time,a)


class GWResponse:
    def __init__(self,h,vertices,g,sigma1,sigma_tau,mu,grid,beta):
        self.h,self.l=np.asarray(h),np.asarray(vertices)
        self.g,self.s1,self.st=np.asarray(g),np.asarray(sigma1),np.asarray(sigma_tau)
        self.mu,self.ir=mu,IRTransforms(grid,beta)
        self.ns,self.n=g.shape[1],g.shape[-1]
        self.spin_weight=2 if self.ns==1 else 1
        if g.shape!=(len(self.ir.tau),self.ns,self.n,self.n):
            raise ValueError('Unexpected checkpoint dimensions')
        if np.max(abs(self.l-self.l.conj().swapaxes(-1,-2)))>1e-10:
            raise ValueError('Coulomb vertices must be Hermitian')
        self.sw=self.ir.f_to_w(self.st)
        self.gw=self.dyson(self.sw,self.ir.fomega)
        self.pi=self.polarization(self.g)
        self.pw=self.ir.b_to_w(self.pi)
        eye=np.eye(len(self.l))
        self.ww=np.linalg.inv(eye-self.pw)
        self.wct=self.ir.w_to_btau(self.ww-eye)
        self.rho=-self.g[-1]
        self.fixed_point=dict(
            green=float(np.max(abs(self.ir.w_to_f(self.gw)-self.g))),
            static_selfenergy=float(np.max(abs(self.hf(self.rho)-self.s1))),
            dynamic_selfenergy=float(np.max(abs(self.dynamic(self.g,self.wct)-self.st))))
        self._setups={}
        self._initialize_transfer_basis()
        if max(self.fixed_point.values())>1e-7:
            raise ValueError('The input checkpoint or basis reconstruction is not a sufficiently converged electronic GW fixed point: '+str(self.fixed_point))

    def _initialize_transfer_basis(self):
        import irbasis
        from numpy.polynomial.legendre import leggauss
        self.fb=irbasis.load('F',self.ir.lambda_)
        self.bb=irbasis.load('B',self.ir.lambda_)
        # Native grids have a few additional tiny singular functions beyond
        # the packaged database. Fit the shared basis to native time data and
        # quantify the discarded/checkpoint reconstruction error explicitly.
        ut=basis_tau(self.fb,self.ir.fx,self.ir.beta)
        native_tau=self.g[1:-1]
        self.gc=transform(np.linalg.pinv(ut),native_tau)
        self.sc=transform(np.linalg.pinv(ut),self.st[1:-1])
        self.wc=transform(np.linalg.pinv(self.bb.compute_unl(self.ir.bn)*np.sqrt(self.ir.beta)),
                          self.ww-np.eye(len(self.l)))
        self.fixed_point['transfer_basis_green_fit']=float(np.max(abs(transform(ut,self.gc)-native_tau)))
        self.fixed_point['transfer_basis_sigma_fit']=float(np.max(abs(transform(ut,self.sc)-self.st[1:-1])))
        self.x=np.unique(np.r_[-1.,self.ir.fx,self.ir.bx,leggauss(256)[0],1.])
        self.tau=(self.x+1)*self.ir.beta/2
        self.gt=transform(basis_tau(self.fb,self.x,self.ir.beta),self.gc)
        self.wt=transform(basis_tau(self.bb,self.x,self.ir.beta),self.wc)

    def dyson(self, sigma_w, frequencies):
        return np.linalg.inv((1j*frequencies[:,None,None,None]+self.mu)*np.eye(self.n)-
                             self.h[None,None]-self.s1[None]-sigma_w)

    def polarization(self,g):
        return self.polarization_pair(g,g[::-1])

    def polarization_pair(self,left,right):
        # Explicit order avoids a greedy four-operand path with expensive
        # auxiliary-pair/orbital outer products for continuum DF inputs.
        lg=np.einsum('Qab,tsbc->tsQac',self.l,left,optimize=True)
        rg=np.einsum('Rcd,tsda->tsRca',self.l,right,optimize=True)
        return -self.spin_weight*np.einsum('tsQac,tsRca->tQR',lg,rg,optimize=True)

    def hf(self,rho):
        charge=self.spin_weight*np.einsum('Qij,sji->Q',self.l,rho)
        hartree=np.einsum('Q,Qij->ij',charge,self.l)
        exchange=np.einsum('Qij,sjk,Qkl->sil',self.l,rho,self.l,optimize=True)
        return hartree[None]-exchange

    def dynamic(self,g,wct):
        lg=np.einsum('Qim,tsmn->tsQin',self.l,g,optimize=True)
        screened=np.einsum('tQR,tsQin->tsRin',wct,lg,optimize=True)
        return -np.einsum('tsRin,Rnj->tsij',screened,self.l,optimize=True)

    def setup(self, transfer_index):
        if not isinstance(transfer_index,(int,np.integer)) or transfer_index<0:
            raise ValueError('Use a nonnegative integer bosonic Matsubara transfer')
        if transfer_index in self._setups:
            return self._setups[transfer_index]
        nu=2*np.pi*transfer_index/self.ir.beta
        phase=np.exp(1j*nu*self.tau)
        fmaps=TransferMaps(self.fb,self.ir.fn,self.x,self.ir.beta,transfer_index)
        bmaps=TransferMaps(self.bb,self.ir.bn,self.x,self.ir.beta,transfer_index)
        fvalues=self.fb.compute_unl(np.r_[fmaps.n,fmaps.n+transfer_index])*np.sqrt(self.ir.beta)
        sw=transform(fvalues[:len(fmaps.n)],self.sc)
        shifted_sw=transform(fvalues[len(fmaps.n):],self.sc)
        fomega=(2*fmaps.n+1)*np.pi/self.ir.beta
        gw=self.dyson(sw,fomega)
        g_shift=self.dyson(shifted_sw,fomega+nu)
        bvalues=self.bb.compute_unl(np.r_[bmaps.n,bmaps.n+transfer_index])*np.sqrt(self.ir.beta)
        ww=np.eye(len(self.l))+transform(bvalues[:len(bmaps.n)],self.wc)
        w_shift=np.eye(len(self.l))+transform(bvalues[len(bmaps.n):],self.wc)
        result=dict(nu=nu,phase=phase,g_w=gw,g_shift=g_shift,w_w=ww,w_shift=w_shift,
                    fmaps=fmaps,bmaps=bmaps)
        self._setups[transfer_index]=result
        return result

    def sigma_variation(self,delta_g_w,setup,screening=True,hartree_exchange=True,dynamic=True):
        dg=setup['fmaps'].backward(delta_g_w)
        drho=-dg[-1]
        ds1=self.hf(drho) if hartree_exchange else np.zeros_like(self.s1)
        dst=self.dynamic(dg,self.wt) if dynamic else np.zeros_like(self.gt)
        if screening and dynamic:
            dp=self.polarization_pair(dg,self.gt[::-1])+setup['phase'][:,None,None]*self.polarization_pair(self.gt,dg[::-1])
            dpw=setup['bmaps'].forward(dp)
            dw=setup['w_shift']@dpw@setup['w_w']
            dwt=setup['bmaps'].backward(dw)
            dst+=self.dynamic(self.gt,dwt)
        return ds1[None]+setup['fmaps'].forward(dst)

    def solve(self,operator,transfer_index,variant='conserving',rtol=2e-11,maxiter=30,initial=None):
        """Return -delta<O>/delta u for delta H=u O exp(-i nu tau).

        conserving: full derivative; fixed_W: omit delta W; bubble: freeze all
        self-energies; Hartree: dress the bubble with Hartree only (native RPA
        comparison). These choices are controls, not interchangeable responses.
        """
        started=time.monotonic()
        operator=np.asarray(operator)
        setup=self.setup(transfer_index)
        shape=setup['g_w'].shape
        b=setup['g_shift']@operator[None,None]@setup['g_w']
        trace=[]

        def delta_sigma(x):
            if variant=='bubble':
                return np.zeros_like(x)
            if variant=='Hartree':
                drho=-setup['fmaps'].backward(x)[-1]
                charge=self.spin_weight*np.einsum('Qij,sji->Q',self.l,drho)
                return np.einsum('Q,Qij->ij',charge,self.l)[None,None]
            if variant not in ('conserving','fixed_W','HF'):
                raise ValueError(variant)
            return self.sigma_variation(x,setup,screening=variant=='conserving',dynamic=variant!='HF')

        def action(flat):
            x=flat.reshape(shape)
            return (x-setup['g_shift']@delta_sigma(x)@setup['g_w']).ravel()

        if variant=='bubble' or not np.any(self.l):
            x,info=b,0
        else:
            a=LinearOperator((b.size,b.size),matvec=action,dtype=complex)
            x,info=gmres(a,b.ravel(),x0=b.ravel() if initial is None else initial.ravel(),
                rtol=rtol,atol=1e-14,restart=30,maxiter=maxiter,
                callback=lambda v:trace.append(float(v)),callback_type='pr_norm')
            x=x.reshape(shape)
        residual=float(np.linalg.norm(action(x.ravel())-b.ravel())/(np.linalg.norm(b)+1e-30))
        drho=-setup['fmaps'].backward(x)[-1]
        chi=-self.spin_weight*np.einsum('ij,sji->',operator,drho)
        diagnostics=dict(transfer_index=int(transfer_index),nu=float(setup['nu']),variant=variant,
            gmres_info=int(info),iterations=len(trace),linear_relative_residual=residual,
            susceptibility_real=float(chi.real),susceptibility_imag=float(chi.imag),
            total_density_variation_real=float((self.spin_weight*np.trace(drho,axis1=-2,axis2=-1).sum()).real),
            transfer_fermi_rank=setup['fmaps'].rank,transfer_bose_rank=setup['bmaps'].rank,
            elapsed_seconds=time.monotonic()-started)
        if transfer_index:
            ward=(setup['g_w']-setup['g_shift'])/(1j*setup['nu'])
            if np.max(abs(operator-np.eye(self.n)))<1e-12:
                diagnostics['ward_solution_relative_error']=float(np.linalg.norm(x-ward)/(np.linalg.norm(ward)+1e-30))
                diagnostics['ward_kernel_residual']=float(np.linalg.norm(action(ward.ravel())-b.ravel())/(np.linalg.norm(b)+1e-30))
        return diagnostics,drho,x

    def static_derivative_check(self,direction,epsilon=1e-4):
        """Central finite difference of the full stationary GW functional."""
        setup=self.setup(0)
        native_basis=basis_tau(self.fb,self.ir.fx,self.ir.beta)
        coefficients=transform(np.linalg.pinv(native_basis),direction[1:-1])
        dg=transform(basis_tau(self.fb,self.x,self.ir.beta),coefficients)
        x=setup['fmaps'].forward(dg)
        dg=setup['fmaps'].backward(x)

        def functional(g):
            pw=setup['bmaps'].forward(self.polarization(g))
            wc=np.linalg.inv(np.eye(len(self.l))-pw)-np.eye(len(self.l))
            return self.hf(-g[-1])[None]+setup['fmaps'].forward(self.dynamic(g,setup['bmaps'].backward(wc)))

        fd=(functional(self.gt+epsilon*dg)-functional(self.gt-epsilon*dg))/(2*epsilon)
        derivative=self.sigma_variation(x,setup)
        fixed_w=self.sigma_variation(x,setup,screening=False)
        scale=np.linalg.norm(derivative)+1e-30
        return dict(full_relative_error=float(np.linalg.norm(fd-derivative)/scale),
                    fixed_W_relative_error=float(np.linalg.norm(fd-fixed_w)/scale),epsilon=epsilon)

    def projected_rpa(self,operator):
        """Native internal bubble plus Coulomb screening, independent of BSE."""
        vertices=np.concatenate((self.l,operator[None]))
        pi=-self.spin_weight*np.einsum('Qab,tsbc,Rcd,tsda->tQR',
            vertices,self.g,vertices,self.g[::-1],optimize=True)
        pw=self.ir.b_to_w(pi)
        nq=len(self.l)
        full=pw[:,-1,-1]+np.einsum('wQ,wQR,wR->w',pw[:,-1,:nq],self.ww,pw[:,:nq,-1])
        return -full

    def photon_coefficients(self,operator,quadratic,chi_native_bose,omega):
        """Weak-coupling free-energy/ground-state coefficients; no feedback.

        H_int=eta Q O+eta² Q² K, hence Delta E=eta² coefficient+O(eta^4).
        The finite-beta trace is a free-energy coefficient; a cold, gapped
        reference is required when interpreting it as a ground-state energy.
        """
        rho=self.spin_weight*self.rho.sum(axis=0)
        mean=float(np.trace(rho@operator).real)
        kappa=float(np.trace(rho@quadratic).real)
        c0=2*omega/(self.ir.bomega**2+omega**2)
        weighted=chi_native_bose*c0
        variance_weighted=chi_native_bose*c0*c0
        integral=self.ir.w_to_btau(weighted[:,None,None])[0,0,0]
        variance_integral=self.ir.w_to_btau(variance_weighted[:,None,None])[0,0,0]
        return dict(energy_coefficient=float((kappa-mean*mean/omega-.5*integral).real),
            coherent_coefficient=-mean*mean/omega,quadratic_coefficient=kappa,
            connected_coefficient=float((-.5*integral).real),
            covariance_Q_coefficient=float((-2*kappa/omega+variance_integral).real),
            frequency_sum_imag=float(abs(integral.imag)),operator_mean=mean)


def exact_density_reference(h,pair,constant,nelectron,operator,quadratic,frequencies,omega=1.):
    """Independent fixed-N electronic Lehmann spectrum and weak-photon limit."""
    from spatial_cavity import electronic_basis,lift_onebody
    basis,lookup=electronic_basis(len(h),nelectron)
    occ=np.array([[((s>>i)&1)+((s>>(len(h)+i))&1) for i in range(len(h))] for s in basis])
    diagonal=.5*(np.einsum('bi,ij,bj->b',occ,pair,occ)-occ@np.diag(pair))+constant
    matrix=lift_onebody(h,basis,lookup).toarray()+np.diag(diagonal)
    e,u=np.linalg.eigh(matrix)
    op=lift_onebody(operator,basis,lookup).toarray()
    quad=lift_onebody(quadratic,basis,lookup).toarray()
    amplitudes=u.conj().T@op@u[:,0]
    gaps=e[1:]-e[0]
    weights=abs(amplitudes[1:])**2
    mean=float(amplitudes[0].real)
    kappa=float(np.vdot(u[:,0],quad@u[:,0]).real)
    chi=np.array([sum(2*gaps*weights/(gaps*gaps+nu*nu)) for nu in frequencies])
    return dict(energy_coefficient=float(kappa-mean*mean/omega-sum(weights/(gaps+omega))),
        operator_mean=mean,quadratic_coefficient=kappa,
        covariance_Q_coefficient=float(-2*kappa/omega+sum(2*weights*(1/(omega*(omega+gaps))+1/(omega+gaps)**2))),
        susceptibility=chi.tolist(),electronic_energy=float(e[0]),gaps=gaps.tolist(),weights=weights.tolist())


def lehmann_susceptibility(h,operator,mu,beta,nu):
    """Independent noninteracting response, including static thermal diagonal."""
    from scipy.special import expit
    e,u=np.linalg.eigh(h)
    op=u.conj().T@operator@u
    f=expit(-beta*(e-mu))
    result=0j
    for i in range(len(e)):
        for j in range(len(e)):
            gap=e[j]-e[i]
            if abs(gap)<1e-12 and abs(nu)<1e-12:
                result+=2*beta*f[i]*(1-f[i])*abs(op[i,j])**2
            elif abs(gap-1j*nu)>1e-14:
                result+=2*(f[i]-f[j])*abs(op[i,j])**2/(gap-1j*nu)
    return result
