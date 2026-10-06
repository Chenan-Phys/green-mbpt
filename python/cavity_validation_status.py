#!/usr/bin/env python3
"""Keep implementation checks and physical readiness separate in one report."""
import argparse
import hashlib
import json
from pathlib import Path


def main(args):
    paths={name:Path(getattr(args,name)).resolve() for name in
           ('limits','workflow','invalid_inputs','oracle','physical')}
    reports={name:json.loads(path.read_text()) for name,path in paths.items()}
    limits=reports['limits']['implementation_checks']
    workflow=reports['workflow']['checks']; invalid=reports['invalid_inputs']
    oracle=reports['oracle']
    groups={'analytic_limits':{'passed':sum(x['passed'] for x in limits.values()),'total':len(limits)},
            'workflow':{'passed':sum(x['passed'] for x in workflow.values()),'total':len(workflow)},
            'invalid_inputs':{'passed':sum(x['passed'] for x in invalid.values()),'total':len(invalid)},
            'independent_tensor_oracle_cases':{'passed':sum(x['implementation_passed'] for x in oracle),
                                               'total':len(oracle)}}
    implementation_passed=all(x['total']>0 and x['passed']==x['total'] for x in groups.values())
    diagnostics=reports['physical']['diagnostics']
    physical_passed=bool(diagnostics['physical_origin_gate']['passed'] and
                         diagnostics['photon_cutoff_passed'])
    output={'implementation':{'passed':implementation_passed,'groups':groups},
            'physical_readiness':{'passed':physical_passed,'diagnostics':diagnostics,
                'interpretation':'A failed physical gate is retained even if implementation regressions pass'},
            'evidence':{name:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
                        for name,path in paths.items()},
            'require_physical':args.require_physical,
            'exit_status':0 if implementation_passed and (physical_passed or not args.require_physical) else 1}
    Path(args.output).write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))
    return output['exit_status']


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in ('limits','workflow','invalid-inputs','oracle','physical'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--require-physical',action='store_true',
                        help='Return nonzero when physical readiness fails, while retaining the report')
    raise SystemExit(main(parser.parse_args()))
