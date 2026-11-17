"""Conserving q_y=0 solve in the strip's invariant orbital-block subspace.

Only momentum-diagonal matrix elements are unknowns. The number mode is retained
and tested; this is not a charge projection or a Ward-identity replacement.
"""
import time
import numpy as np
from scipy.sparse.linalg import LinearOperator,gmres


def solve(engine,operator,transfer_index,block_size,variant='conserving',rtol=2e-11,maxiter=30,initial=None,preconditioner=True):
    started=time.monotonic();operator=np.asarray(operator);setup=engine.setup(transfer_index);n=engine.n;nk=n//block_size
    if operator.shape!=(n,n):raise ValueError('This solver accepts a common orbital probe; spin-dependent fields require a separate spin-probe interface')
    ids=np.arange(n).reshape(nk,block_size)
    def unpack(a):return a[...,ids[:,:,None],ids[:,None,:]]
    def pack(a):
        out=np.zeros(a.shape[:-3]+(n,n),complex)
        for k in range(nk):out[...,ids[k,:,None],ids[k,None,:]]=a[...,k,:,:]
        return out
    gw,shift=unpack(setup['g_w']),unpack(setup['g_shift']);op=unpack(operator)
    b=shift@op[None,None]@gw;shape=b.shape;history=[]
    def delta_sigma(x):
        if variant=='bubble':return np.zeros_like(x)
        full=pack(x)
        if variant=='Hartree':
            rho=-setup['fmaps'].backward(full)[-1]
            charge=engine.spin_weight*np.einsum('Qij,sji->Q',engine.l,rho)
            value=np.einsum('Q,Qij->ij',charge,engine.l)
            return unpack(value)[None,None]
        if variant not in ('conserving','fixed_W','HF'):raise ValueError(variant)
        return unpack(engine.sigma_variation(full,setup,screening=variant=='conserving',dynamic=variant!='HF'))
    def action(flat):
        x=flat.reshape(shape)
        return (x-shift@delta_sigma(x)@gw).ravel()
    precondition=None;condition=None
    if preconditioner and np.any(engine.l) and variant!='bubble':
        # Exact instantaneous HF inverse as a numerical preconditioner. The
        # solved operator and final residual still include the full dynamic
        # functional derivative. Every charge/spin mode is retained.
        if not hasattr(engine,'_translation_hf_kernel'):
            basis=[]
            for a in range(block_size):
                value=np.zeros((block_size,block_size),complex);value[a,a]=1.;basis.append(value)
            for a in range(block_size):
                for b0 in range(a+1,block_size):
                    value=np.zeros((block_size,block_size),complex);value[a,b0]=value[b0,a]=1/np.sqrt(2);basis.append(value)
                    value=np.zeros_like(value);value[a,b0]=-1j/np.sqrt(2);value[b0,a]=1j/np.sqrt(2);basis.append(value)
            basis=np.array(basis);rank=len(basis);dimension=engine.ns*nk*rank
            rho_basis=np.zeros((dimension,engine.ns,nk,block_size,block_size),complex)
            for j in range(dimension):
                s,k,p=np.unravel_index(j,(engine.ns,nk,rank));rho_basis[j,s,k]=basis[p]
            full_basis=pack(rho_basis);v=engine.v if hasattr(engine,'v') else np.einsum('Qab,Qcd->abcd',engine.l,engine.l,optimize=True)
            direct=v.reshape(n*n,n*n);exchange=v.transpose(0,3,1,2).reshape(n*n,n*n)
            charge=engine.spin_weight*full_basis.sum(axis=1).swapaxes(-1,-2).reshape(dimension,-1)
            j=(direct@charge.T).T.reshape(dimension,n,n)
            k=np.stack([(exchange@full_basis[:,s].reshape(dimension,-1).T).T.reshape(dimension,n,n) for s in range(engine.ns)],axis=1)
            kernel=np.einsum('pab,Dskba->Dskp',basis,unpack(j[:,None]-k),optimize=True).reshape(dimension,dimension).T
            engine._translation_hf_kernel=(basis,kernel)
        basis,kernel=engine._translation_hf_kernel;rank=len(basis);dimension=len(kernel)
        left=np.einsum('pab,wskbc->wskpac',basis,shift,optimize=True)
        right=np.einsum('pab,wskbc->wskpac',basis,gw,optimize=True)
        chi=-np.einsum('wskpab,wskqba,w->skpq',left,right,setup['fmaps'].to_time[-1],optimize=True)
        susceptibility=np.zeros_like(kernel)
        for s in range(engine.ns):
            for k in range(nk):
                offset=(s*nk+k)*rank;susceptibility[offset:offset+rank,offset:offset+rank]=chi[s,k]
        reduced=np.eye(dimension)-susceptibility@kernel;condition=float(np.linalg.cond(reduced))
        if condition<1e10:
            effective=kernel@np.linalg.inv(reduced)
            def apply(flat):
                r=flat.reshape(shape);rho=-np.einsum('w,wskab->skab',setup['fmaps'].to_time[-1],r,optimize=True)
                coeff=np.einsum('pab,skba->skp',basis,rho,optimize=True).reshape(-1)
                field=np.einsum('skp,pab->skab',(effective@coeff).reshape(engine.ns,nk,rank),basis,optimize=True)
                return (r+shift@field[None]@gw).ravel()
            precondition=LinearOperator((b.size,b.size),matvec=apply,dtype=complex)
    if variant=='bubble' or not np.any(engine.l):x,info=b,0
    else:
        a=LinearOperator((b.size,b.size),matvec=action,dtype=complex)
        x,info=gmres(a,b.ravel(),M=precondition,x0=b.ravel() if initial is None else unpack(initial).ravel(),rtol=rtol,atol=1e-14,restart=30,maxiter=maxiter,
            callback=lambda v:history.append(float(v)),callback_type='pr_norm');x=x.reshape(shape)
    residual=float(np.linalg.norm(action(x.ravel())-b.ravel())/(np.linalg.norm(b)+1e-30))
    full=pack(x);drho=-setup['fmaps'].backward(full)[-1];chi=-engine.spin_weight*np.einsum('ij,sji->',operator,drho)
    diagnostics=dict(transfer_index=int(transfer_index),nu=float(setup['nu']),variant=variant,gmres_info=int(info),iterations=len(history),linear_relative_residual=residual,
        susceptibility_real=float(chi.real),susceptibility_imag=float(chi.imag),total_density_variation_real=float((engine.spin_weight*np.trace(drho,axis1=-2,axis2=-1).sum()).real),
        transfer_fermi_rank=setup['fmaps'].rank,transfer_bose_rank=setup['bmaps'].rank,elapsed_seconds=time.monotonic()-started,representation='Exact longitudinal q=0 orbital blocks')
    diagnostics.update(HF_preconditioned=precondition is not None,HF_preconditioner_condition=condition)
    if transfer_index and np.max(abs(operator-np.eye(n)))<1e-12:
        ward=(gw-shift)/(1j*setup['nu'])
        diagnostics['ward_solution_relative_error']=float(np.linalg.norm(x-ward)/(np.linalg.norm(ward)+1e-30))
        diagnostics['ward_kernel_residual']=float(np.linalg.norm(action(ward.ravel())-b.ravel())/(np.linalg.norm(b)+1e-30))
    return diagnostics,drho,full
