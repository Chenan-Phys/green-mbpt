"""External response to independent up/down one-body fields.

Uses the supplied conserving GW/GF2 engine's derivative, without changing its
electronic approximation. This also permits static longitudinal spin stability
tests of spin-symmetric stationary states. Global dynamic N_up/N_down Ward
controls apply; a static response need not vanish.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres


def solve(engine,operators,transfer_index,rtol=2e-11,maxiter=6):
    operators=np.asarray(operators)
    if operators.shape!=(engine.ns,engine.n,engine.n): raise ValueError('One matrix per stored spin required')
    started=time.monotonic();setup=engine.setup(transfer_index);shape=setup['g_w'].shape
    b=setup['g_shift']@operators[None]@setup['g_w'];history=[]
    def action(flat):
        x=flat.reshape(shape)
        return (x-setup['g_shift']@engine.sigma_variation(x,setup)@setup['g_w']).ravel()
    if not np.any(engine.l): x,info=b,0
    else:
        a=LinearOperator((b.size,b.size),matvec=action,dtype=complex)
        x,info=gmres(a,b.ravel(),x0=b.ravel(),rtol=rtol,atol=1e-14,restart=30,maxiter=maxiter,
            callback=lambda v:history.append(float(v)),callback_type='pr_norm')
        x=x.reshape(shape)
    drho=-setup['fmaps'].backward(x)[-1]
    chi=-engine.spin_weight*np.einsum('sij,sji->',operators,drho)
    residual=np.linalg.norm(action(x.ravel())-b.ravel())/(np.linalg.norm(b)+1e-30)
    return dict(transfer_index=int(transfer_index),susceptibility_real=float(chi.real),susceptibility_imag=float(chi.imag),
        gmres_info=int(info),iterations=len(history),linear_relative_residual=float(residual),
        total_number_derivative=float((engine.spin_weight*np.trace(drho,axis1=-2,axis2=-1).sum()).real),
        elapsed_seconds=time.monotonic()-started),drho
