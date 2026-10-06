#!/usr/bin/env python3
"""Export a scalar molecular PF input using installed stable MBtools (run on workstation)."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import h5py
import numpy as np
from pyscf import gto, lib
BOHR = lib.param.BOHR
from green_mbtools.mint import common_utils
from green_mbtools.mint.pyscf_init import pyscf_mol_init


class ChargedMolecularExporter(pyscf_mol_init):
    """Stable exporter has no charge CLI option; override only molecular construction."""
    def __init__(self, args, charge):
        self.charge = int(charge)
        super().__init__(args)

    def cell_object(self):
        mol = gto.M(atom=self.args.atom, basis=self.args.basis, ecp=self.args.ecp,
                    unit="Angstrom", charge=self.charge, spin=self.args.spin,
                    verbose=4, max_memory=int(os.environ.get("PYSCF_MAX_MEMORY", "4000")))
        return mol

    def mf_object(self, mydf=None):
        if self.args.auxbasis is None:
            return super().mf_object(mydf)
        # Stable solve_mol_mean_field ignores its mydf/auxbasis argument. Its
        # cached factors then disagree with the exporter's selected NQ. Keep
        # this correction in the adapter; do not modify the installed package.
        if self.args.x2c != 0 or self.args.xc is not None:
            raise NotImplementedError("Explicit auxiliary basis currently supports scalar HF")
        mf=self.args.mean_field(self.cell).density_fit(auxbasis=self.args.auxbasis)
        mf.with_df._cderi_to_save="cderi_mol.h5"
        mf.with_df.build()
        mf.diis_space=16
        mf.damp=self.args.damping
        mf.max_cycle=self.args.max_iter
        mf.chkfile="tmp.chk"
        mf.kernel()
        mf.analyze()
        return mf


def prepare(spec, directory):
    if spec.get("native_basis","ao")!="ao":
        raise ValueError("Export the AO input, then use cavity_lowdin.py for a verified full-basis conversion")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "input.h5").exists():
        raise FileExistsError(f"Refusing to overwrite {directory / 'input.h5'}")
    atoms = spec["atoms"]
    if not atoms or any(len(atom) != 4 for atom in atoms):
        raise ValueError("atoms must contain [symbol, x, y, z] rows in Angstrom")
    geometry = "; ".join(f"{a[0]} {a[1]:.10f} {a[2]:.10f} {a[3]:.10f}" for a in atoms)
    charge, spin = int(spec.get("charge", 0)), int(spec.get("spin", 0))
    basis = spec.get("basis", "def2-svp")
    argv = ["--atom", geometry, "--basis", basis, "--spin", str(spin),
            "--restricted", str(spin == 0).lower(), "--orth", "none",
            "--memory", "500", "--keep_cderi", "true", "--max_iter", "150"]
    if spec.get("ecp"):
        argv += ["--ecp", spec["ecp"]]
    if spec.get("auxbasis"):
        argv += ["--auxbasis", spec["auxbasis"]]
    args = common_utils.init_mol_params(argv)
    previous = Path.cwd()
    try:
        os.chdir(directory)
        exporter = ChargedMolecularExporter(args, charge)
        exporter.mean_field_input()
        mol = exporter.cell
        if mol.nelectron != sum(gto.charge(mol.atom_symbol(i))-mol.atom_nelec_core(i)
                                for i in range(mol.natm))-charge:
            raise ValueError("Unexpected ECP electron bookkeeping")
        pol = np.asarray(spec.get("polarization", [0., 0., 1.]), float)
        if pol.shape != (3,) or not np.all(np.isfinite(pol)) or np.linalg.norm(pol) == 0:
            raise ValueError("Invalid cavity polarization")
        pol /= np.linalg.norm(pol)
        origin = np.asarray(spec.get("origin_angstrom", [0., 0., 0.]), float) / BOHR
        mol.set_common_orig(origin)
        r = mol.intor("int1e_r", comp=3)
        rr = mol.intor("int1e_rr", comp=9).reshape(3, 3, mol.nao_nr(), mol.nao_nr())
        d = -np.einsum("a,aij->ij", pol, r)
        second = np.einsum("a,b,abij->ij", pol, pol, rr)
        ionic = np.asarray([gto.charge(mol.atom_symbol(i))-mol.atom_nelec_core(i)
                            for i in range(mol.natm)], float)
        mu_nuc = np.einsum("i,ia,a->", ionic, mol.atom_coords()-origin, pol)
        with h5py.File("input.h5", "a") as handle:
            q = handle.create_group("QED")
            q["schema"] = np.int32(1)
            q["molecular"] = np.int32(1)
            q["omega_hartree"] = float(spec.get("omega_ev", 2.0)) / 27.211386245988
            q["lambda_au"] = float(spec.get("lambda_au", 0.0))
            q["nuclear_dipole"] = mu_nuc
            q["dipole"] = d
            q["second_moment"] = second
            q["polarization"] = pol
            q["origin_bohr"] = origin
            q["effective_ionic_charges"] = ionic
            q["ao_atom_index"] = np.asarray([int(label[0]) for label in mol.ao_labels(fmt=False)])
            q["geometry_angstrom"] = mol.atom_coords()*BOHR
            if spec.get("fix_spin",False):
                if spin==0:
                    raise ValueError("Fixed-spin input requires an unrestricted molecular specification")
                q.create_group("fixed_spin")["target"]=np.asarray(mol.nelec,dtype=float)
        Path("molecule.json").write_text(mol.dumps())
        provenance = dict(spec, nelectron=mol.nelectron, nao=mol.nao_nr(),
                          ionic_charges=ionic.tolist(), mbtools_file=sys.modules[pyscf_mol_init.__module__].__file__,
                          input_sha256=hashlib.sha256(Path("input.h5").read_bytes()).hexdigest())
        Path("manifest.json").write_text(json.dumps(provenance, indent=2)+"\n")
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("directory")
    options = parser.parse_args()
    prepare(json.loads(Path(options.spec).read_text()), options.directory)
