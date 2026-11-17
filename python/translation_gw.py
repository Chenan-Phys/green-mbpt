"""Conserving GW response with independent longitudinal Coulomb channels.

For a y-invariant strip and q_y=0 external probe, polarization is block diagonal
in the magnitude of Coulomb momentum transfer. Cosine/sine partners remain in
the same block, including their complex magnetic response. No time-even or
bare-vertex approximation is introduced. Native electronic GW is unchanged.
"""
import numpy as np
from conserving_response import GWResponse,IRTransforms,TransferMaps,basis_tau,transform


def auxiliary_groups(vertices,orbital_block_size,tolerance=1e-10):
    n=vertices.shape[-1];p=np.arange(n)//orbital_block_size
    difference=abs(p[:,None]-p[None,:]);groups={}
    for i,l in enumerate(vertices):
        support=abs(l)>tolerance*max(float(np.max(abs(l))),1e-20)
        transfer=np.unique(difference[support])
        if len(transfer)!=1:raise ValueError('DF vertex mixes longitudinal transfer sectors')
        groups.setdefault(int(transfer[0]),[]).append(i)
    return [np.array(v) for k,v in sorted(groups.items())]


def pair(vertices,left,right,spin_weight,time_chunk=16):
    result=np.empty((len(left),len(vertices),len(vertices)),complex)
    for start in range(0,len(left),time_chunk):
        sl=slice(start,start+time_chunk)
        lg=np.einsum('Qab,tsbc->tsQac',vertices,left[sl],optimize=True)
        rg=np.einsum('Rcd,tsda->tsRca',vertices,right[sl],optimize=True)
        result[sl]=-spin_weight*np.einsum('tsQac,tsRca->tQR',lg,rg,optimize=True)
    return result


def dynamic_selfenergy(vertices,g,w,time_chunk=16):
    out=np.empty_like(g,dtype=complex)
    for start in range(0,len(g),time_chunk):
        sl=slice(start,start+time_chunk)
        lg=np.einsum('Qim,tsmn->tsQin',vertices,g[sl],optimize=True)
        screened=np.einsum('tQR,tsQin->tsRin',w[sl],lg,optimize=True)
        out[sl]=-np.einsum('tsRin,Rnj->tsij',screened,vertices,optimize=True)
    return out


class BlockGWResponse(GWResponse):
    def __init__(self,h,vertices,g,sigma1,sigma_tau,mu,grid,beta,orbital_block_size):
        self.h,self.l=np.asarray(h),np.asarray(vertices);self.g,self.s1,self.st=np.asarray(g),np.asarray(sigma1),np.asarray(sigma_tau)
        self.mu,self.ir=mu,IRTransforms(grid,beta);self.ns,self.n=g.shape[1],g.shape[-1];self.spin_weight=2 if self.ns==1 else 1
        if self.h.shape not in ((self.n,self.n),(self.ns,self.n,self.n)):raise ValueError('Unexpected one-body shape')
        if np.max(abs(self.l-self.l.conj().swapaxes(-1,-2)))>1e-10:raise ValueError('Hermitian DF vertices required')
        p=np.arange(self.n)//orbital_block_size;self.off=p[:,None]!=p[None,:]
        self.orbital_block_size=orbital_block_size
        self.check(self.g);self.check(self.h);self.check(self.s1);self.check(self.st)
        self.sw=self.ir.f_to_w(self.st);self.gw=self.dyson(self.sw,self.ir.fomega);self.rho=-g[-1]
        self.groups=auxiliary_groups(self.l,orbital_block_size);self.channels=[];reconstructed=np.zeros_like(self.st,dtype=complex)
        import irbasis
        from numpy.polynomial.legendre import leggauss
        self.fb=irbasis.load('F',self.ir.lambda_);self.bb=irbasis.load('B',self.ir.lambda_)
        ut=basis_tau(self.fb,self.ir.fx,beta);self.gc=transform(np.linalg.pinv(ut),g[1:-1]);self.sc=transform(np.linalg.pinv(ut),self.st[1:-1])
        self.x=np.unique(np.r_[-1.,self.ir.fx,self.ir.bx,leggauss(256)[0],1.]);self.tau=(self.x+1)*beta/2
        self.gt=transform(basis_tau(self.fb,self.x,beta),self.gc)
        inverse=np.linalg.pinv(self.bb.compute_unl(self.ir.bn)*np.sqrt(beta));bt=basis_tau(self.bb,self.x,beta)
        for indices in self.groups:
            ls=self.l[indices];pw=self.ir.b_to_w(pair(ls,g,g[::-1],self.spin_weight));eye=np.eye(len(ls));ww=np.linalg.inv(eye-pw)
            wc=self.ir.w_to_btau(ww-eye);reconstructed+=dynamic_selfenergy(ls,g,wc)
            coeff=transform(inverse,ww-eye)
            self.channels.append(dict(vertices=ls,coefficients=coeff,time=transform(bt,coeff)))
        self.fixed_point=dict(green=float(np.max(abs(self.ir.w_to_f(self.gw)-g))),static_selfenergy=float(np.max(abs(self.hf(self.rho)-self.s1))),
            dynamic_selfenergy=float(np.max(abs(reconstructed-self.st))),transfer_basis_green_fit=float(np.max(abs(transform(ut,self.gc)-g[1:-1]))),
            transfer_basis_sigma_fit=float(np.max(abs(transform(ut,self.sc)-self.st[1:-1]))))
        self._setups={}
        if max(self.fixed_point.values())>1e-7:raise ValueError('Not a converged native GW fixed point: '+str(self.fixed_point))

    def check(self,a):
        if np.max(abs(a[...,self.off]),initial=0.)>1e-10*(1+np.max(abs(a),initial=0.)):
            raise ValueError('Probe/propagator breaks longitudinal translation')

    def setup(self,m):
        if not isinstance(m,(int,np.integer)) or m<0:raise ValueError('Nonnegative integer transfer required')
        if m in self._setups:return self._setups[m]
        nu=2*np.pi*m/self.ir.beta;fm=TransferMaps(self.fb,self.ir.fn,self.x,self.ir.beta,m);bm=TransferMaps(self.bb,self.ir.bn,self.x,self.ir.beta,m)
        values=self.fb.compute_unl(np.r_[fm.n,fm.n+m])*np.sqrt(self.ir.beta);count=len(fm.n);omega=(2*fm.n+1)*np.pi/self.ir.beta
        bv=self.bb.compute_unl(np.r_[bm.n,bm.n+m])*np.sqrt(self.ir.beta);w=[];shift=[]
        for c in self.channels:
            eye=np.eye(len(c['vertices']));w.append(eye+transform(bv[:len(bm.n)],c['coefficients']));shift.append(eye+transform(bv[len(bm.n):],c['coefficients']))
        result=dict(nu=nu,phase=np.exp(1j*nu*self.tau),fmaps=fm,bmaps=bm,g_w=self.dyson(transform(values[:count],self.sc),omega),
            g_shift=self.dyson(transform(values[count:],self.sc),omega+nu),channel_w=w,channel_shift=shift)
        self._setups[m]=result;return result

    def sigma_variation(self,x,setup,screening=True,hartree_exchange=True,dynamic=True):
        self.check(x);dg=setup['fmaps'].backward(x)
        ds1=self.hf(-dg[-1]) if hartree_exchange else np.zeros_like(self.s1);ds=np.zeros_like(dg)
        if dynamic:
            for i,c in enumerate(self.channels):
                l=c['vertices'];ds+=dynamic_selfenergy(l,dg,c['time'])
                if screening:
                    dp=pair(l,dg,self.gt[::-1],self.spin_weight)+setup['phase'][:,None,None]*pair(l,self.gt,dg[::-1],self.spin_weight)
                    dw=setup['channel_shift'][i]@setup['bmaps'].forward(dp)@setup['channel_w'][i]
                    ds+=dynamic_selfenergy(l,self.gt,setup['bmaps'].backward(dw))
        return ds1[None]+setup['fmaps'].forward(ds)

    def solve(self,operator,*args,**kwargs):
        self.check(np.asarray(operator))
        from translation_response import solve
        return solve(self,operator,*args,block_size=self.orbital_block_size,**kwargs)

    def static_derivative_check(self,direction,epsilon=1e-4):
        setup=self.setup(0);ut=basis_tau(self.fb,self.ir.fx,self.ir.beta)
        dg=transform(basis_tau(self.fb,self.x,self.ir.beta),transform(np.linalg.pinv(ut),direction[1:-1]));x=setup['fmaps'].forward(dg);dg=setup['fmaps'].backward(x)
        def functional(g):
            ds=np.zeros_like(g)
            for c in self.channels:
                l=c['vertices'];eye=np.eye(len(l));pw=setup['bmaps'].forward(pair(l,g,g[::-1],self.spin_weight))
                ds+=dynamic_selfenergy(l,g,setup['bmaps'].backward(np.linalg.inv(eye-pw)-eye))
            return self.hf(-g[-1])[None]+setup['fmaps'].forward(ds)
        finite=(functional(self.gt+epsilon*dg)-functional(self.gt-epsilon*dg))/(2*epsilon)
        exact=self.sigma_variation(x,setup);scale=np.linalg.norm(exact)+1e-30
        return dict(full_relative_error=float(np.linalg.norm(finite-exact)/scale),fixed_W_relative_error=float(np.linalg.norm(finite-self.sigma_variation(x,setup,screening=False))/scale),epsilon=epsilon)

    def projected_rpa(self,operator):
        raise NotImplementedError('Use the full conserving response; this representation does not retain a global internal W array')
