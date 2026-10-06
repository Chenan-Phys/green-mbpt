#!/usr/bin/env python3
"""Independent NumPy tensor contractions of a saved native G and joint screening.

Uses the recorded IR tables; it tests tensor/spin/screening/observable assembly,
not the mathematical validity of the bare-bubble GW approximation itself.
"""
import argparse
import json
from pathlib import Path
import socket
import h5py
import numpy as np
from cavity_analyze import complex_array


def transforms(path,beta):
    with h5py.File(path) as handle:
        a=np.sqrt(2/beta); b=np.sqrt(beta/2)
        def stats(name):
            ux=np.asarray(handle[name+'/uxl']); uw=np.asarray(handle[name+'/uwl'])
            tc=np.vstack((handle[name+'/u1l_neg'],ux,handle[name+'/u1l_pos']))*a
            ct=np.linalg.pinv(ux)*b
            nc=uw*np.sqrt(beta); cn=np.linalg.pinv(uw)/np.sqrt(beta)
            return tc,ct,nc,cn
        ft,fc,fn,fni=stats('fermi'); bt,bc,bn,bni=stats('bose')
        other=bt@bc@np.asarray(handle['fermi/uxl_other'])*a
        inverse_other=ft@fc@np.asarray(handle['bose/uxl_other'])*a
        forward=bn@bc@other[1:-1]@fc
        backward=inverse_other@bni
        frequencies=2*np.pi*np.asarray(handle['bose/ngrid'])/beta
    return forward,backward,frequencies


def check(directory):
    if socket.gethostname().split('.')[0]!='kadanoff': raise RuntimeError('Workstation only')
    directory=Path(directory).resolve()
    if not directory.is_relative_to(Path('/data/cwei/green_runs').resolve()):
        raise ValueError('Requires task data')
    stamp=json.loads((directory/'gw_provenance.json').read_text())
    beta=float(stamp['beta_hartree_inverse']); command=stamp['command']
    grid=Path(command[command.index('--grid_file')+1])
    forward,backward,frequencies=transforms(grid,beta)
    with h5py.File(directory/'input.h5') as handle:
        dipole=np.asarray(handle['QED/dipole']); coupling=float(handle['QED/lambda_au'][()])
        overlap=complex_array(handle['HF/S-k'])[0,0]
        omega=float(handle['QED/omega_hartree'][()])
    with h5py.File(directory/'df_hf_int/VQ_0.h5') as handle:
        stored=np.asarray(handle['0'])
        if stored.dtype.kind!='c':
            stored=np.ascontiguousarray(stored).view(np.complex128)
        coulomb=stored[0]
        if coulomb.shape[1:]!=dipole.shape: raise ValueError('Unexpected native vertex layout')
    vertices=np.concatenate((coulomb,coupling*dipole[None]))
    all_vertices=np.concatenate((vertices,overlap[None]))
    with h5py.File(directory/'gw.h5') as handle:
        iteration=max(int(k[4:]) for k in handle if k.startswith('iter') and k[4:].isdigit())
        group=handle['iter'+str(iteration)]
        g=complex_array(group['G_tau/data'])[:,:,0]
        actual_chi=complex_array(group['QED/Chi_dipole_w']).reshape(-1)
        actual_delta=complex_array(group['QED/Photon_delta_w']).reshape(-1)
        actual_variance=float(group['QED/Photon_variance'][()])
        actual_energy=float(group['QED/Photon_energy_correction'][()])
        actual_sigma=complex_array(group['Selfenergy/data'])[:,:,0]
    ns=g.shape[1]; weight=2 if ns==1 else 1
    p_tau=np.zeros((len(g),len(all_vertices),len(all_vertices)),complex)
    for t in range(len(g)//2):
        # Independent four-index trace contraction, without the native GEMM reshapes.
        value=-weight*np.einsum('Qab,sbc,Rcd,sda->QR',all_vertices,g[t],
            all_vertices.conj(),g[-1-t],optimize=True)
        value=.5*(value+value.conj().T)
        p_tau[t]=value; p_tau[-1-t]=value
    p_w=np.einsum('nt,tQR->nQR',forward,p_tau[1:-1],optimize=True)
    expected_chi=[]; number_chi=[]; correlation_w=[]
    for nu,full in zip(frequencies,p_w):
        p=full[:-1,:-1]
        weights=np.ones(len(vertices)); weights[-1]=abs(nu)/np.sqrt(nu**2+omega**2)
        bare=np.diag(weights**2)
        # Direct W = (I-BP)^-1 B, independently of the native symmetric LDLT route.
        screened=np.linalg.solve(np.eye(len(vertices))-bare@p,bare)
        correlation_w.append(.5*(screened+screened.conj().T)-np.eye(len(vertices)))
        chi=p+p@screened@p
        expected_chi.append(chi[-1,-1])
        number_chi.append(full[-1,-1]+full[-1,:-1]@screened@full[:-1,-1])
    expected_chi=np.asarray(expected_chi); number_chi=np.asarray(number_chi)
    denominator=frequencies**2+omega**2
    delta=2*omega**3*expected_chi/denominator**2
    energy_w=omega**2*frequencies**2*expected_chi/denominator**2
    nb=1/np.expm1(beta*omega)
    variance=1+2*nb-(backward@delta)[-1].real
    energy=omega*nb+(backward@energy_w)[-1].real
    correlation_tau=np.einsum('tn,nQR->tQR',backward,np.asarray(correlation_w),optimize=True)
    expected_sigma=-np.einsum('Qim,tsmn,Rjn,tQR->tsij',vertices,g,vertices.conj(),
                             correlation_tau,optimize=True)
    errors={'chi_w_max_abs':float(np.max(np.abs(expected_chi-actual_chi))),
            'photon_delta_w_max_abs':float(np.max(np.abs(delta-actual_delta))),
            'photon_variance_abs':float(abs(variance-actual_variance)),
            'photon_energy_abs_hartree':float(abs(energy-actual_energy)),
            'selfenergy_tau_fixed_point_max_abs':float(np.max(np.abs(expected_sigma-actual_sigma)))}
    output={'directory':str(directory),'iteration':iteration,'errors':errors,
        'implementation_passed':all(v<1e-10 for v in errors.values()),
        'dynamic_number_response_max_abs':float(np.max(np.abs(number_chi[np.abs(frequencies)>1e-12]))),
        'interpretation':'The independently contracted internal GW bubble/ring response must match native output. Its finite-frequency response to conserved N should vanish in the exact theory; that physical criterion is separate.'}
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('directory',nargs='+'); parser.add_argument('--output',required=True)
    args=parser.parse_args(); result=[check(p) for p in args.directory]
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not all(p['implementation_passed'] for p in result): raise AssertionError('Frozen-G oracle mismatch')
