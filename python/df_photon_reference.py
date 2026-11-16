"""Small fixed-spin ED references for Hermitian-DF Coulomb Hamiltonians.

H = dGamma(h) + 1/2 sum_Q [dGamma(L_Q)^2-dGamma(L_Q^2)] + c.
No GW equations enter this reference. Weak-photon energy and density are
computed with electronic resolvents, without truncating excited-state sums.
Scientific callers must arrange a compute allocation before importing.
"""
import numpy as np
from scipy.sparse import eye,kron,diags,coo_matrix
from scipy.sparse.linalg import eigsh,cg,LinearOperator
from spatial_cavity import electronic_basis,lift_onebody


class DFReference:
    def __init__(self,h,vertices,constant,nelectron,singlet=False):
        self.n=len(h)
        if np.max(abs(vertices-vertices.conj().swapaxes(1,2)))>1e-10:
            raise ValueError('Hermitian DF vertices required')
        self.basis,self.lookup=electronic_basis(self.n,nelectron)
        self.hamiltonian=lift_onebody(h,self.basis,self.lookup)+constant*eye(len(self.basis))
        for vertex in vertices:
            op=self.lift(vertex)
            self.hamiltonian+=.5*(op@op-self.lift(vertex@vertex))
        self.hamiltonian=self.hamiltonian.tocsr()
        # S+ maps out of minimal Sz. Build S²=S-S+ + Sz(Sz+1), including
        # fermionic signs, so a requested singlet comparison is explicit.
        nup,ndown=(nelectron+1)//2,nelectron//2
        sz=(nup-ndown)/2
        targets={};rows=[];cols=[];values=[]
        for col,state in enumerate(self.basis):
            for a in range(self.n):
                source=self.n+a
                if not (state>>source)&1 or (state>>a)&1: continue
                removed=state^(1<<source)
                sign=(-1)**((state&((1<<source)-1)).bit_count()+(removed&((1<<a)-1)).bit_count())
                target=removed|(1<<a)
                if target not in targets: targets[target]=len(targets)
                rows.append(targets[target]);cols.append(col);values.append(sign)
        raising=coo_matrix((values,(rows,cols)),shape=(len(targets),len(self.basis))).tocsr()
        spin2=raising.conj().T@raising+sz*(sz+1)*eye(len(self.basis))
        self.spin_operator=spin2
        self.spin_constraint='S=0' if singlet else 'minimal Sz; total spin unrestricted'
        if singlet:
            if nelectron%2: raise ValueError('Even N required for singlet')
            # Scalar operators used below preserve total S, so the penalty has
            # no effect inside the singlet resolvents or photon-Fock problem.
            self.hamiltonian+=100.*spin2
        anti=self.hamiltonian-self.hamiltonian.conj().T
        if np.max(abs(anti.data),initial=0)>1e-9: raise RuntimeError('Non-Hermitian ED Hamiltonian')
        rng=np.random.default_rng(6847)
        e,u=eigsh(self.hamiltonian,k=2,which='SA',tol=1e-12,v0=rng.normal(size=len(self.basis)))
        order=np.argsort(e);self.energy=float(e[order[0]]);self.ground=u[:,order[0]]
        self.gap=float(e[order[1]]-self.energy)
        if self.gap<1e-7: raise ValueError('Nondegenerate reference required')
        self.shifted=self.hamiltonian-self.energy*eye(len(self.basis))
        self.residual=float(np.linalg.norm(self.shifted@self.ground))
        if self.residual>1e-8: raise RuntimeError('ED ground residual too large')
        # Minimal Sz contains all allowed total-spin sectors, not just singlets.
        # Diagnose spontaneous polarization before comparing to restricted GW.
        self.spin_squared=float(np.vdot(self.ground,spin2@self.ground).real)
        if singlet and abs(self.spin_squared)>1e-7: raise RuntimeError('Singlet penalty did not select S=0')

    def lift(self,matrix):
        return lift_onebody(matrix,self.basis,self.lookup)

    def weak_photon(self,current,quadratic,omega):
        """Coefficient C and delta-rho/eta^2 for eta Q J + eta^2 Q^2 K.

        Density includes normalization, one-photon and second-order zero-photon
        amplitudes. Uses fixed N_up,N_down; no charge projection on delta-rho.
        """
        j,k=self.lift(current),self.lift(quadratic)
        ground=self.ground
        source=j@ground
        resolvent=self.shifted+omega*eye(len(ground))
        u,info=cg(resolvent,-source,rtol=2e-12,atol=1e-14,maxiter=2000)
        if info: raise RuntimeError('One-photon solve failed')
        c=float(np.vdot(ground,k@ground).real+np.vdot(source,u).real)
        norm=float(np.vdot(u,u).real)
        s=k@ground+j@u
        s-=ground*np.vdot(ground,s)
        projected=LinearOperator(self.shifted.shape,matvec=lambda v:self.shifted@v+ground*np.vdot(ground,v),dtype=complex)
        v,info=cg(projected,-s,rtol=2e-12,atol=1e-14,maxiter=2000)
        if info: raise RuntimeError('Zero-photon solve failed')
        rho=np.zeros((self.n,self.n),complex);delta=np.zeros_like(rho)
        for a in range(self.n):
            for b in range(self.n):
                matrix=np.zeros_like(rho);matrix[a,b]=1.
                op=self.lift(matrix)
                rho[b,a]=np.vdot(ground,op@ground)
                delta[b,a]=np.vdot(u,op@u)-norm*rho[b,a]+np.vdot(ground,op@v)+np.vdot(v,op@ground)
        return dict(energy_coefficient=c,gap=self.gap,electronic_energy=self.energy,
            spin_squared=self.spin_squared,
            spin_constraint=self.spin_constraint,
            electronic_dimension=len(ground),eigen_residual=self.residual,
            operator_mean=float(np.vdot(ground,source).real),quadratic_coefficient=float(np.vdot(ground,k@ground).real),
            one_photon_residual=float(np.linalg.norm(resolvent@u+source)),
            zero_photon_residual=float(np.linalg.norm(projected@v+s)),
            density_charge_coefficient=float(np.trace(delta).real),
            density_hermiticity=float(np.max(abs(delta-delta.conj().T))),
            rho=rho,delta_rho_coefficient=delta)

    def coupled(self,current,quadratic,omega,eta,cutoff=5):
        """Independent full photon-Fock diagonalization for limiting tests."""
        a=diags(np.sqrt(np.arange(1,cutoff)),1,shape=(cutoff,cutoff));q=a+a.T
        ham=kron(self.hamiltonian,eye(cutoff))+kron(eye(len(self.basis)),omega*diags(np.arange(cutoff,dtype=float)))
        ham+=eta*kron(self.lift(current),q)+eta**2*kron(self.lift(quadratic),q@q)
        e,u=eigsh(ham,k=1,which='SA',tol=1e-12,v0=np.random.default_rng(9464).normal(size=ham.shape[0]))
        psi=u[:,0].reshape(len(self.basis),cutoff)
        rho=np.zeros((self.n,self.n),complex)
        for a in range(self.n):
            for b in range(self.n):
                matrix=np.zeros_like(rho);matrix[a,b]=1.
                rho[b,a]=np.sum(psi.conj()*(self.lift(matrix)@psi))
        return float(e[0]),rho
