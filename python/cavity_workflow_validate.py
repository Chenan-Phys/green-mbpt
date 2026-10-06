#!/usr/bin/env python3
"""Exercise cache rejection and mandatory native warm-start reconvergence."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import socket
import h5py

from cavity_limits import prepare_model
from cavity_run import run
from cavity_warm_start import seed
from cavity_analyze import analyze


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(source,root):
    if socket.gethostname().split('.')[0]!='kadanoff':
        raise RuntimeError('Scientific validation must run on GREEN_workstation')
    source=Path(source).resolve(); root=Path(root).resolve()
    allowed=Path('/data/cwei/green_runs/qed-validation').resolve()
    if not source.is_relative_to(allowed) or not root.is_relative_to(allowed):
        raise ValueError('Validation copies must stay under task validation data')
    root.mkdir(parents=True,exist_ok=False)
    checks={}

    def save():
        output={'checks':checks,'all_passed':all(x['passed'] for x in checks.values()),
                'interpretation':'Private copies; accepted native source and production files remain unchanged'}
        (root/'workflow_validation.json').write_text(json.dumps(output,indent=2)+'\n')
        return output

    def copy(name):
        target=root/name
        shutil.copytree(source,target)
        return target

    cached=copy('valid_cache')
    before=sha(cached/'gw.h5')
    result=run(cached/'manifest.json',cached,methods=('GW',),energy_threshold=1e-12)
    checks['valid_cache']={'passed':before==sha(cached/'gw.h5'),
                           'iteration':result['GW']['iteration']}
    save()
    for name,message in [('input','Input changed'),('input_force_restart','Input changed'),
            ('binary','different executable/core'),('core','different executable/core'),
            ('grid','different frequency/time grid'),('temperature','different temperature'),
            ('ensemble','Cached ensemble differs'),('no_provenance','no provenance'),
            ('legacy_wrong_grid','grid cannot be verified')]:
        target=copy(name)
        stamp_path=target/'gw_provenance.json'
        stamp=json.loads(stamp_path.read_text())
        if name.startswith('input'):
            with h5py.File(target/'input.h5','r+') as handle:
                handle['HF/Energy_nuc'][()]=float(handle['HF/Energy_nuc'][()])+1e-6
        elif name in ('binary','core','grid'):
            key={'binary':'binary_sha256','core':'core_source_sha256_lf','grid':'grid_sha256'}[name]
            stamp[key]='0'*64
        elif name=='temperature': stamp['beta_hartree_inverse']=500.
        elif name=='ensemble': stamp['const_density']=False
        elif name=='legacy_wrong_grid':
            stamp.pop('grid_sha256',None)
            stamp['grid_file']='/not-the-original-grid.h5'
        if name=='no_provenance': stamp_path.unlink()
        else: stamp_path.write_text(json.dumps(stamp,indent=2)+'\n')
        checkpoint=sha(target/'gw.h5')
        caught=''
        try:
            run(target/'manifest.json',target,methods=('GW',),energy_threshold=1e-12,
                force_restart=name=='input_force_restart')
        except RuntimeError as exc:
            caught=str(exc)
        checks[name]={'passed':message in caught and checkpoint==sha(target/'gw.h5'),
                      'exception':caught,'expected_message':message}
        save()

    source_digest=sha(source/'gw.h5')
    for legacy,increment in [(False,1e-9),(True,2e-9)]:
        name='legacy_seed' if legacy else 'new_seed'
        target=root/name
        coupling=json.loads((source/'manifest.json').read_text())['lambda_au']+increment
        spec=prepare_model(source,target,coupling=coupling)
        record=seed(source,target)
        stamp_path=target/'gw_provenance.json'
        if legacy:
            stamp=json.loads(stamp_path.read_text()); stamp.pop('requires_reconvergence')
            stamp_path.write_text(json.dumps(stamp,indent=2)+'\n')
        seeded=analyze(target,'gw.h5')
        # Deliberately choose a tiny lambda step whose transferred solution
        # already passes all postprocessing tolerances: it must still iterate.
        value=run(spec,target,methods=('GW',),energy_threshold=1e-12,
                  number_tolerance=1e-13,mixing_weight=.7)['GW']
        stamp=json.loads(stamp_path.read_text())
        native_ran=('--restart' in value['command'] and
                    value['iteration']>seeded['iteration'] and
                    not stamp.get('requires_reconvergence') and
                    'initial_guess' not in stamp and
                    stamp.get('previous_run_provenance',{}).get('initial_guess',{}).get('seeded'))
        checks[name]={'passed':bool(record['seeded'] and seeded['accepted_numerically'] and
                    native_ran and value['accepted_numerically'] and source_digest==sha(source/'gw.h5')),
                    'lambda_step':increment,'seed_passed_postprocessing':seeded['accepted_numerically'],
                    'seed_iteration':seeded['iteration'],'target_iteration':value['iteration'],
                    'source_checkpoint_unchanged':source_digest==sha(source/'gw.h5')}
        save()
    output=save()
    print(json.dumps(output,indent=2))
    if not output['all_passed']: raise AssertionError('Workflow regression failed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('source'); parser.add_argument('root')
    args=parser.parse_args(); main(args.source,args.root)
