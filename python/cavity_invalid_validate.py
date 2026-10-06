#!/usr/bin/env python3
"""Ensure explicitly unsupported QED inputs fail for their intended reason."""
import argparse
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import h5py
import numpy as np


def main(source,root):
    if socket.gethostname().split('.')[0]!='kadanoff': raise RuntimeError('Workstation only')
    source=Path(source).resolve(); root=Path(root).resolve()
    base=Path('/data/cwei/green_runs/qed-validation').resolve()
    if not source.is_relative_to(base) or not root.is_relative_to(base): raise ValueError('Task validation paths only')
    root.mkdir(parents=True,exist_ok=False)
    tests=[('gf2','GF2',None,'QED input currently supports HF'),
        ('zero_frequency','HF','omega','Nonfinite, non-Hermitian or invalid QED'),
        ('nonfinite_coupling','HF','nan','Nonfinite, non-Hermitian or invalid QED'),
        ('asymmetric_dipole','HF','dipole','Nonfinite, non-Hermitian or invalid QED'),
        ('nonmolecular','HF','molecular','QED schema 1 requires'),
        ('wrong_spin_count','HF','spin','Fixed-spin populations must sum')]
    table={}
    for name,method,change,expected in tests:
        target=root/name; target.mkdir()
        shutil.copy2(source/'input.h5',target/'input.h5')
        (target/'df_hf_int').symlink_to((source/'df_hf_int').resolve(),target_is_directory=True)
        with h5py.File(target/'input.h5','r+') as handle:
            if change=='omega': handle['QED/omega_hartree'][()]=0.
            if change=='nan': handle['QED/lambda_au'][()]=np.nan
            if change=='dipole': handle['QED/dipole'][0,1]+=1.
            if change=='molecular': handle['QED/molecular'][()]=0
            if change=='spin': handle['QED/fixed_spin/target'][...]=0.
        command=[str(Path.home()/'green/install-qed/cavity-general-density/bin/mbpt.exe'),
            '--scf_type',method,'--kernel','CPU','--BETA','1000',
            '--grid_file',str(Path.home()/'green/install/share/ir/1e5.h5'),
            '--dfintegral_file','df_hf_int','--results_file','expected_failure.h5']
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        result=subprocess.run(command,cwd=target,env=env,stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT,text=True,timeout=30)
        (target/'native.log').write_text(result.stdout)
        table[name]={'returncode':result.returncode,'expected_message':expected,
            'message_found':expected in result.stdout,'passed':result.returncode!=0 and expected in result.stdout}
    (root/'invalid_input_validation.json').write_text(json.dumps(table,indent=2)+'\n')
    print(json.dumps(table,indent=2))
    if not all(x['passed'] for x in table.values()): raise AssertionError('An unsupported input was not rejected correctly')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('root')
    args=parser.parse_args(); main(args.source,args.root)
