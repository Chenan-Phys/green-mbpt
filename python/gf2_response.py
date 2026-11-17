"""Finite orthonormal, collinear GF2 self-energy and conserving response.

Independent orbital-tensor contractions for auditing stable native GF2. The
response differentiates all three electron lines and Hartree/Fock, with the
same two-leg Matsubara representation as conserving_response.GWResponse.
Requires an externally arranged compute allocation before scientific imports.
"""
from types import SimpleNamespace
import numpy as np
from conserving_response import GWResponse,IRTransforms,TransferMaps,transform


def coulomb_tensor(vertices):
    return np.einsum('Qab,Qcd->abcd',vertices,vertices,optimize=True)


def cubic(g1,g2,g3,v,spin_weight=1):
    """Trilinear second Born functional; g3 is evaluated at beta-tau.

    Direct: V_ibac G^s_bd V_djef G^r_ce G^r_fa.
    Exchange: -V_ibac G^s_bd V_dfej G^s_ce G^s_fa.
    v_abcd multiplies a^dagger_a a^dagger_c a_d a_b in H/2.
    """
    out=np.zeros_like(g1,dtype=complex)
    for s in range(g1.shape[1]):
        left=np.einsum('ibac,tbd->tiacd',v,g1[:,s],optimize=True)
        for r in range(g2.shape[1]):
            mid=np.einsum('tiacd,tce->tiade',left,g2[:,r],optimize=True)
            three=np.einsum('tiade,tfa->tifde',mid,g3[:,r],optimize=True)
            out[:,s]+=spin_weight*np.einsum('tifde,djef->tij',three,v,optimize=True)
            if s==r:
                out[:,s]-=np.einsum('tifde,dfej->tij',three,v,optimize=True)
    return out


class GF2Response(GWResponse):
    def __init__(self,h,vertices,g,sigma1,sigma_tau,mu,grid,beta):
        self.h,self.l=np.asarray(h),np.asarray(vertices)
        self.g,self.s1,self.st=np.asarray(g),np.asarray(sigma1),np.asarray(sigma_tau)
        self.mu,self.ir=mu,IRTransforms(grid,beta)
        self.ns,self.n=g.shape[1],g.shape[-1]
        self.spin_weight=2 if self.ns==1 else 1
        if g.shape!=(len(self.ir.tau),self.ns,self.n,self.n): raise ValueError('Unexpected checkpoint dimensions')
        if np.max(abs(self.l-self.l.conj().swapaxes(-1,-2)))>1e-10: raise ValueError('Hermitian DF required')
        self.v=coulomb_tensor(self.l)
        self.sw=self.ir.f_to_w(self.st);self.gw=self.dyson(self.sw,self.ir.fomega);self.rho=-self.g[-1]
        self.fixed_point=dict(green=float(np.max(abs(self.ir.w_to_f(self.gw)-self.g))),
            static_selfenergy=float(np.max(abs(self.hf(self.rho)-self.s1))),
            dynamic_selfenergy=float(np.max(abs(cubic(g,g,g[::-1],self.v,self.spin_weight)-self.st))))
        # Reuse the well-tested equilibrium IR fits; screened-interaction arrays
        # are zero placeholders and never enter GF2's self-energy or response.
        self.ww=np.broadcast_to(np.eye(len(self.l)),(len(self.ir.bn),len(self.l),len(self.l)))
        self._setups={};self._initialize_transfer_basis()
        if max(self.fixed_point.values())>1e-7: raise ValueError('Not a converged native GF2 fixed point: '+str(self.fixed_point))

    def setup(self,transfer_index):
        if not isinstance(transfer_index,(int,np.integer)) or transfer_index<0: raise ValueError('Nonnegative integer transfer required')
        if transfer_index in self._setups: return self._setups[transfer_index]
        nu=2*np.pi*transfer_index/self.ir.beta
        fm=TransferMaps(self.fb,self.ir.fn,self.x,self.ir.beta,transfer_index)
        values=self.fb.compute_unl(np.r_[fm.n,fm.n+transfer_index])*np.sqrt(self.ir.beta)
        nw=len(fm.n);omega=(2*fm.n+1)*np.pi/self.ir.beta
        result=dict(nu=nu,phase=np.exp(1j*nu*self.tau),fmaps=fm,bmaps=SimpleNamespace(rank=0),
            g_w=self.dyson(transform(values[:nw],self.sc),omega),
            g_shift=self.dyson(transform(values[nw:],self.sc),omega+nu))
        self._setups[transfer_index]=result
        return result

    def sigma_variation(self,delta_g_w,setup,screening=True,hartree_exchange=True,dynamic=True):
        dg=setup['fmaps'].backward(delta_g_w)
        ds1=self.hf(-dg[-1]) if hartree_exchange else np.zeros_like(self.s1)
        ds=np.zeros_like(dg)
        if dynamic:
            args=(self.v,self.spin_weight)
            ds=cubic(dg,self.gt,self.gt[::-1],*args)+cubic(self.gt,dg,self.gt[::-1],*args)
            ds+=setup['phase'][:,None,None,None]*cubic(self.gt,self.gt,dg[::-1],*args)
        return ds1[None]+setup['fmaps'].forward(ds)

    def solve(self,operator,transfer_index,variant='conserving',**kwargs):
        if variant not in ('conserving','bubble','Hartree','HF'): raise ValueError('Unsupported GF2 response variant')
        return super().solve(operator,transfer_index,variant=variant,**kwargs)

    def static_derivative_check(self,direction,epsilon=1e-4):
        setup=self.setup(0);dg=setup['fmaps'].backward(direction)
        def functional(g):
            return self.hf(-g[-1])[None]+setup['fmaps'].forward(cubic(g,g,g[::-1],self.v,self.spin_weight))
        fd=(functional(self.gt+epsilon*dg)-functional(self.gt-epsilon*dg))/(2*epsilon)
        exact=self.sigma_variation(direction,setup)
        return float(np.linalg.norm(fd-exact)/(np.linalg.norm(exact)+1e-30))

    def projected_rpa(self,operator):
        raise NotImplementedError('A GW internal screened bubble is not a GF2 response')
