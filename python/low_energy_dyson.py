"""Low-frequency Matsubara Dyson estimates, with explicit fit-window diagnostics.

This is a quasiparticle linearization, not unique analytic continuation. It
accepts spin-resolved one-body terms, including Zeeman splitting. Scientific
work must be arranged in a compute allocation by the calling application.
"""
import numpy as np


def estimate(h,sigma1,sigma_iw,omega,mu,points=6):
    h=np.asarray(h);sigma1=np.asarray(sigma1);sigma_iw=np.asarray(sigma_iw);omega=np.asarray(omega)
    ids=np.flatnonzero(omega>0)[:points]
    if len(ids)<3:raise ValueError('At least three positive frequencies required')
    w=omega[ids];values=sigma_iw[ids]
    herm=(values+values.conj().swapaxes(-1,-2))/2
    anti=(values-values.conj().swapaxes(-1,-2))/(2j)
    even=np.c_[np.ones_like(w),w*w]
    odd=np.c_[np.ones_like(w),w]
    real_fit=np.linalg.lstsq(even,herm.reshape(len(w),-1),rcond=None)[0]
    imag_fit=np.linalg.lstsq(odd,anti.reshape(len(w),-1),rcond=None)[0]
    shape=sigma1.shape;shift=real_fit[0].reshape(shape);slope=imag_fit[1].reshape(shape)
    scattering=-imag_fit[0].reshape(shape)
    metric=np.eye(shape[-1])-slope
    metric=(metric+metric.conj().swapaxes(-1,-2))/2
    eig,u=np.linalg.eigh(metric)
    if eig.min()<=0:raise ValueError('Nonpositive low-frequency quasiparticle metric')
    inverse=(u*(1/np.sqrt(eig))[...,None,:])@u.conj().swapaxes(-1,-2)
    h0=h+sigma1+shift-mu*np.eye(shape[-1])
    qp=inverse@h0@inverse;qp=(qp+qp.conj().swapaxes(-1,-2))/2
    return dict(hamiltonian=qp,metric=metric,scattering=scattering,
        real_fit_residual=float(np.max(abs((even@real_fit).reshape(herm.shape)-herm))),
        imag_fit_residual=float(np.max(abs((odd@imag_fit).reshape(anti.shape)-anti))),
        min_metric_eigenvalue=float(eig.min()),min_scattering_eigenvalue=float(np.linalg.eigvalsh(scattering).min()),
        max_frequency=float(w.max()),points=len(ids),scope='Low-frequency Matsubara quasiparticle estimate, not a real-frequency spectrum')
