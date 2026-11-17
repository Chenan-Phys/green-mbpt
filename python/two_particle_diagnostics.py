"""Small collinear GW/GF2 thermodynamic spin/number diagnostics.

Uses the disconnected correlated-density product plus the approximate cumulant,
via the functional-derivative strategy of Pokhilko et al., JCP 155, 024119 (2021),
arXiv:2104.10308. These are
approximate two-particle diagnostics; they need not be ensemble representable.
For GW the operator-index convention of that paper matters because its cumulant
is not antisymmetric. GW uses the explicit exchanged-annihilator operator
representation checked by the full pair-space Phi derivative in the application.
Equivalent fermionic operator representations can differ for its non-antisymmetric
cumulant; the returned value is a convention-dependent diagnostic, not a total-spin
projection. GF2 uses antisymmetrized interaction vertices.
"""
import numpy as np
from conserving_response import basis_tau,transform
from gf2_response import coulomb_tensor


def spin_matrices(n):
    eye=np.eye(n);z=np.zeros_like(eye)
    return [np.block([[z,eye],[eye,z]])/2,
            np.block([[z,-1j*eye],[1j*eye,z]])/2,
            np.block([[eye,z],[z,-eye]])/2]


def spinor(a):
    if a.shape[1]==1: a=np.repeat(a,2,axis=1)
    n=a.shape[-1];out=np.zeros((len(a),2*n,2*n),complex)
    out[:,:n,:n]=a[:,0];out[:,n:,n:]=a[:,1]
    return out


def antisym(vertices):
    v=coulomb_tensor(vertices)
    return v.transpose(0,2,1,3)-v.transpose(0,2,3,1)


def gf2_cumulant(w,g,back,operators):
    # Equivalent to the partial derivative of Phi_2 at fixed G under
    # delta v_abcd = 2 sum_A O_A,ab O_A,cd. The four closed electron lines
    # reduce to two M=G(tau) O G(beta-tau) matrices after antisymmetrization.
    out=np.zeros(len(g),complex)
    for op in operators:
        m=g@op@back
        out-=np.einsum('iabc,tbi,tca->t',w,m,m,optimize=True)
    return out


def evaluate(engine,method,gauss_order=256):
    n=engine.n;rho=engine.rho
    if engine.ns==1: rho=np.repeat(rho,2,axis=0)
    density=spinor(rho[None])[0];ops=spin_matrices(n);number=np.trace(density).real
    wick_spin=sum((np.trace(density@op@op)+np.trace(density@op)**2-np.trace(density@op@density@op)).real for op in ops)
    wick_var=float((np.trace(density)-np.trace(density@density)).real)
    if method=='GF2':
        from numpy.polynomial.legendre import leggauss
        x,weights=leggauss(gauss_order);weights*=engine.ir.beta/2
        small=transform(basis_tau(engine.fb,x,engine.ir.beta),engine.gc)
        g=spinor(small);back=g[::-1]
        ls=np.array([np.kron(np.eye(2),q) for q in engine.l]);w=antisym(ls)
        spin_integrand=gf2_cumulant(w,g,back,ops)
        number_integrand=gf2_cumulant(w,g,back,[np.eye(2*n)])
        cs=weights@spin_integrand;cn=weights@number_integrand
        from gf2_response import cubic
        sigma=cubic(small,small,small[::-1],engine.v,engine.spin_weight)
        energy=-.5*engine.spin_weight*np.einsum('t,tsij,tsji->',weights,sigma,small[::-1])
    elif method=='GW':
        # D_Q,ij = -(G L_Q G_reverse)_ij; Gamma chemical = -D W D.
        lg=np.einsum('Qab,tsbj->tsQaj',engine.l,engine.g[::-1],optimize=True)
        d=-np.einsum('tsia,tsQaj->tsQij',engine.g,lg,optimize=True)
        dw=engine.ir.b_to_w(d)
        reverse=engine.ir.b_to_w(d[::-1])
        if engine.ns==1:
            dw=np.repeat(dw,2,axis=1);reverse=np.repeat(reverse,2,axis=1)
        same=np.einsum('wsQij,wQR,wsRji->w',dw,engine.ww,reverse,optimize=True)
        cross=np.einsum('wQij,wQR,wRji->w',dw[:,0],engine.ww,reverse[:,1],optimize=True)
        cross+=np.einsum('wQij,wQR,wRji->w',dw[:,1],engine.ww,reverse[:,0],optimize=True)
        total=lambda a:engine.ir.w_to_btau(a[:,None,None])[0,0,0]
        cs=total(.25*same+.5*cross);cn=total(same)
        energy=-.5*total(np.einsum('wQR,wRS,wSQ->w',engine.pw,engine.ww,engine.pw,optimize=True))
    else: raise ValueError(method)
    return dict(spin_squared=float(wick_spin+cs.real),spin_squared_disconnected=float(wick_spin),
        spin_squared_cumulant=float(cs.real),number_variance=float(wick_var+cn.real),
        number_variance_disconnected=wick_var,number_variance_cumulant=float(cn.real),
        correlation_energy_from_cumulant=float(energy.real),max_imag=float(max(abs(cs.imag),abs(cn.imag),abs(energy.imag))),
        number=float(number),scope='Thermodynamic approximate two-particle diagnostic; GW operator convention explicitly specified; not a spin projector or guarantee of representability')
