#!/usr/bin/env python3
"""Analytic limits of the native solver, plus separately labeled readiness gates.

All models are two-orbital synthetic Hamiltonians, not molecular predictions.
The molecule files supply only electron/spin and tensor-layout metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import numpy as np
import h5py
from pyscf import gto
from cavity_run import run
from cavity_validate import exact_reference


def pack(value):
    return np.stack((np.asarray(value).real,np.asarray(value).imag),axis=-1)


def prepare_model(source,target,coupling=.004,omega=.25,transition=1.,constant=0.,
                  second_defect=0.,rotation=0.,electrons=1,restricted=False):
    source=Path(source).resolve(); target=Path(target).resolve()
    root=Path('/data/cwei/green_runs/qed-validation').resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise ValueError('Validation model paths must stay under task validation data')
    if target.exists(): raise FileExistsError(target)
    target.mkdir(parents=True)
    for name in ('input.h5','cderi_mol.h5'):
        shutil.copy2(source/name,target/name)
    shutil.copytree(source/'df_hf_int',target/'df_hf_int')
    mol=gto.loads((source/'molecule.json').read_text())
    if mol.nao_nr()!=2: raise ValueError('Requires the two-orbital H2 template')
    mol.charge=2-electrons; mol.spin=1 if electrons==1 else 0
    (target/'molecule.json').write_text(mol.dumps())
    ns=1 if restricted else 2
    theta=float(rotation); c,s=np.cos(theta),np.sin(theta)
    transform=np.asarray([[c,-s],[s,c]])
    h=transform.T@np.diag([-.25,.25])@transform
    d=transform.T@(constant*np.eye(2)+transition*np.asarray([[0.,1.],[1.,0.]]))@transform
    r2=d@d+second_defect*np.eye(2)
    with h5py.File(target/'input.h5','r+') as handle:
        for name,value in [('H-k',h),('S-k',np.eye(2)),('Fock-k',h)]:
            del handle['HF/'+name]
            handle['HF/'+name]=pack(np.broadcast_to(value,(ns,1,2,2)))
        handle['HF/Energy_nuc'][()]=0.
        handle['params/nel_cell'][()]=electrons
        handle['params/ns'][()]=ns
        handle['QED/lambda_au'][()]=coupling
        handle['QED/omega_hartree'][()]=omega
        handle['QED/nuclear_dipole'][()]=0.
        handle['QED/dipole'][...]=d
        handle['QED/second_moment'][...]=r2
        if 'QED/fixed_spin' in handle: del handle['QED/fixed_spin']
        if ns==2: handle.create_group('QED/fixed_spin')['target']=np.asarray(mol.nelec,float)
    # These are private copied factors; template and production integrals stay intact.
    for path in (target/'cderi_mol.h5',target/'df_hf_int/VQ_0.h5'):
        with h5py.File(path,'r+') as handle:
            datasets=[]
            handle.visititems(lambda name,obj:datasets.append(name) if isinstance(obj,h5py.Dataset) else None)
            for name in datasets: handle[name][...]=0.
    manifest=json.loads((source/'manifest.json').read_text())
    manifest.update(system='synthetic two-level Pauli-Fierz validation model',
        charge=mol.charge,spin=mol.spin,fix_spin=ns==2,nelectron=electrons,
        native_basis='ao',lambda_au=coupling,omega_ev=omega*27.211386245988,
        synthetic_model={'level_gap_hartree':.5,'transition_dipole':transition,
            'constant_dipole':constant,'second_moment_defect':second_defect,
            'rotation_angle_radians':rotation,'electron_coulomb_factors':'zero',
            'interpretation':'tensor/electron-count template only; no molecular interpretation'},
        input_sha256=hashlib.sha256((target/'input.h5').read_bytes()).hexdigest())
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return target/'manifest.json'


def main(source,root):
    if socket.gethostname().split('.')[0]!='kadanoff': raise RuntimeError('Workstation only')
    root=Path(root).resolve(); root.mkdir(parents=True,exist_ok=False)
    table={}; checks={}
    def case(name,beta=1000.,grid=None,**kwargs):
        directory=root/name
        spec=prepare_model(source,directory,**kwargs)
        result=run(spec,directory,beta=beta,energy_threshold=1e-12,
            number_tolerance=1e-13,native_threads=4,mixing_weight=.7,grid_file=grid)
        table[name]=result
        (root/'limiting_cases_partial.json').write_text(json.dumps(table,indent=2)+'\n')
        return result
    def gate(name,error,tolerance,interpretation):
        checks[name]={'error':float(error),'tolerance':float(tolerance),
                      'passed':bool(abs(error)<tolerance),'interpretation':interpretation}
    zero=case('zero',coupling=0.)
    for coupling in (.004,.002,.001):
        name='weak_'+str(coupling)
        value=case(name,coupling=coupling)
        energy_coefficient=(value['GW']['energy_hartree']-zero['GW']['energy_hartree'])/coupling**2
        q_coefficient=(value['GW']['Photon_variance']-1)/coupling**2
        expected_energy=.5/(2*(.5+.25))
        expected_q=(.5+2*.25)/(.5+.25)**2
        gate(name+'_energy_coefficient',energy_coefficient-expected_energy,1e-3,
             'Analytic O(lambda^2) energy coefficient; finite-lambda remainder allowed')
        gate(name+'_photon_coefficient',q_coefficient-expected_q,2e-3,
             'Analytic O(lambda^2) connected coordinate-variance coefficient')
        gate(name+'_hf',value['HF']['energy_hartree']+.25-.5*coupling**2,2e-10,
             'Exact coherent HF energy of the synthetic off-diagonal model')
        exact=exact_reference(root/name,coupling,16,observables=True,use_saved_df=True)
        table[name]['same_hamiltonian_exact']=exact
    plus=table['weak_0.004']
    minus=case('negative',coupling=-.004)
    rotated=case('rotated',rotation=.43)
    for method in ('HF','GW'):
        gate(method+'_parity',minus[method]['energy_hartree']-plus[method]['energy_hartree'],1e-10,
             'Even coupling sign')
        gate(method+'_rotation',rotated[method]['energy_hartree']-plus[method]['energy_hartree'],1e-10,
             'Orthogonal orbital rotation leaves the Hamiltonian spectrum invariant')
    gate('photon_rotation',rotated['GW']['Photon_variance']-plus['GW']['Photon_variance'],1e-9,
         'Connected photon variance is invariant under orbital rotations')
    defect=case('second_moment',second_defect=.3)
    for method in ('HF','GW'):
        gate(method+'_second_moment',defect[method]['energy_hartree']-plus[method]['energy_hartree']-.5*.004**2*.3,
             1e-10,'Additional one-body projected second moment must have its full energy weight')
    scalar=case('number_only',coupling=.02,transition=0.,constant=1.)
    for method in ('HF','GW'):
        gate(method+'_conserved_number_mode',scalar[method]['energy_hartree']-zero[method]['energy_hartree'],1e-9,
             'A constant dipole coupled to fixed N is removed by photon displacement')
    gate('number_only_photon',scalar['GW']['Photon_variance']-1,1e-9,
         'A displaced free vacuum has connected coordinate variance one')
    thermal=case('free_thermal',beta=10.,coupling=.02,transition=0.)
    nb=1/np.expm1(10*.25)
    for method in ('HF','GW'):
        gate(method+'_thermal_photon_energy',thermal[method]['Photon_energy_correction']-.25*nb,1e-12,
             'Decoupled thermal photon energy omega*n_B')
        gate(method+'_thermal_photon_variance',thermal[method]['Photon_variance']-(1+2*nb),1e-12,
             'Decoupled thermal photon coordinate variance 1+2*n_B')
    restricted=case('restricted_two_electrons',electrons=2,restricted=True)
    unrestricted=case('unrestricted_two_electrons',electrons=2)
    for method in ('HF','GW'):
        gate(method+'_spin_representation',restricted[method]['energy_hartree']-unrestricted[method]['energy_hartree'],1e-9,
             'Matched restricted and unrestricted closed-shell representations')
    gate('photon_spin_representation',restricted['GW']['Photon_variance']-unrestricted['GW']['Photon_variance'],1e-9,
         'Spin factor in the polarization bubble')
    alternate=Path.home()/'green/install/share/ir/1e6.h5'
    grid=case('grid_1e6',grid=alternate)
    gate('GW_grid_energy',grid['GW']['energy_hartree']-plus['GW']['energy_hartree'],1e-9,
         'IR grid 1e5 versus 1e6')
    gate('GW_grid_photon',grid['GW']['Photon_variance']-plus['GW']['Photon_variance'],1e-8,
         'IR grid 1e5 versus 1e6 photon equal-time transform')
    output={'cases':table,'implementation_checks':checks,
            'all_implementation_limits_passed':all(v['passed'] for v in checks.values()),
            'interpretation':'Exact limits and O(lambda^2) coefficients; does not assert GW equals finite-coupling FCI or fixes molecular origin dependence'}
    (root/'limiting_cases.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(checks,indent=2))
    if not output['all_implementation_limits_passed']: raise AssertionError('One or more analytic limits failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('root')
    args=parser.parse_args(); main(args.source,args.root)
