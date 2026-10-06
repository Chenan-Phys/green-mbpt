#!/usr/bin/env python3
"""Matched-DF exact references and strict-grid molecular origin diagnostics."""
import argparse
import json
from pathlib import Path
import socket
from cavity_thermal_hf import clone_reference
from cavity_validate import exact_reference
from cavity_run import run


def main(source,root,require_origin_invariance=False):
    if socket.gethostname().split('.')[0]!='kadanoff': raise RuntimeError('Workstation only')
    source=Path(source).resolve(); root=Path(root).resolve()
    root.mkdir(parents=True,exist_ok=False)
    data={}
    for label,coupling in [('zero',0.),('plus',.005),('origin',.005)]:
        original=source/label; target=root/label
        clone_reference(original,target)
        result=run(target/'manifest.json',target,energy_threshold=1e-12,
            number_tolerance=1e-13,native_threads=4,mixing_weight=.7,
            grid_file=Path.home()/'green/install/share/ir/1e6.h5')
        result['exact_df_cut8']=exact_reference(target,coupling,8,observables=True,use_saved_df=True)
        result['exact_df_cut16']=exact_reference(target,coupling,16,observables=True,use_saved_df=True)
        data[label]=result
        (root/'matched_df_partial.json').write_text(json.dumps(data,indent=2)+'\n')
    gw_shift=data['plus']['GW']['energy_hartree']-data['zero']['GW']['energy_hartree']
    exact_shift=data['plus']['exact_df_cut16']['energy_hartree']-data['zero']['exact_df_cut16']['energy_hartree']
    origin={method:data['origin'][method]['energy_hartree']-data['plus'][method]['energy_hartree'] for method in ('HF','GW')}
    origin['exact_df']=data['origin']['exact_df_cut16']['energy_hartree']-data['plus']['exact_df_cut16']['energy_hartree']
    exact_induced=data['plus']['exact_df_cut16']['connected_photon_q_variance']-1
    gw_induced=data['plus']['GW']['Photon_variance']-1
    diagnostics={'gw_cavity_shift_hartree':gw_shift,'exact_df_cavity_shift_hartree':exact_shift,
        'cavity_energy_relative_approximation_error':(gw_shift-exact_shift)/exact_shift,
        'gw_induced_photon_q_variance':gw_induced,'exact_df_induced_photon_q_variance':exact_induced,
        'induced_photon_variance_relative_approximation_error':(gw_induced-exact_induced)/exact_induced,
        'origin_energy_changes_hartree':origin,
        'physical_origin_gate':{'tolerance_hartree':1e-8,'passed':abs(origin['GW'])<1e-8,
            'interpretation':'Physical fixed-charge invariance; bare-vertex ring GW may violate this even when its implementation limits pass'},
        'photon_cutoff_passed':all(max(abs(data[k]['exact_df_cut8'][p]-data[k]['exact_df_cut16'][p])
            for p in data[k]['exact_df_cut8'])<1e-9 for k in data),
        'interpretation':'The same saved DF Hamiltonian is used in GREEN and finite electron-photon CI. Finite-coupling GW approximation errors are not coding-error assertions.'}
    output={'cases':data,'diagnostics':diagnostics}
    (root/'matched_df_validation.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(diagnostics,indent=2))
    if not diagnostics['photon_cutoff_passed']: raise AssertionError('Exact reference cutoff failed')
    if require_origin_invariance and not diagnostics['physical_origin_gate']['passed']:
        raise AssertionError('Physical origin-invariance gate failed; evidence retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('root')
    parser.add_argument('--require-origin-invariance',action='store_true',
                        help='Return failure on the physical gate after saving the diagnostics')
    args=parser.parse_args(); main(args.source,args.root,args.require_origin_invariance)
