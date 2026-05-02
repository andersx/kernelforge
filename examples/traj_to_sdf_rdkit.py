"""Convert an extxyz trajectory to a multi-structure SDF using RDKit.

Steps
-----
1. Load the trajectory with ASE.
2. Build an RDKit molecule from frame 0 and run ``rdDetermineBonds`` to assign
   bond orders (single / double / aromatic).
3. For every subsequent frame, clone the reference molecule, update its
   conformer with the new atomic positions, and align it onto frame 0 using
   the RMSD-minimising alignment in ``rdMolAlign``.
4. Write all aligned frames to a multi-structure SDF.

Usage
-----
    python examples/traj_to_sdf_rdkit.py azobenzene_rff_md.extxyz azobenzene_rff_md_rdkit.sdf
    python examples/traj_to_sdf_rdkit.py azobenzene_rff_md.extxyz  # writes <input>.sdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolAlign
from rdkit.Geometry import Point3D


def atoms_to_rdmol(atoms: object) -> Chem.RWMol:  # type: ignore[type-arg]
    """Build an RDKit RWMol from an ASE Atoms object (positions + atomic numbers)."""
    import numpy as np  # noqa: F401  # used via atoms API

    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atoms))  # type: ignore[arg-type]
    for i, (z, pos) in enumerate(zip(atoms.numbers, atoms.positions)):  # type: ignore[union-attr]
        mol.AddAtom(Chem.Atom(int(z)))
        conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    mol.AddConformer(conf, assignId=True)
    return mol


def update_conformer(mol: Chem.Mol, positions: object) -> Chem.Mol:  # type: ignore[type-arg]
    """Return a copy of *mol* with its conformer replaced by *positions* (Å)."""
    mol = Chem.RWMol(mol)
    conf = mol.GetConformer(0)
    for i, pos in enumerate(positions):  # type: ignore[union-attr]
        conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))
    return mol.GetMol()


def traj_to_sdf(extxyz_path: Path, sdf_path: Path, charge: int = 0) -> None:
    """Convert *extxyz_path* trajectory to aligned multi-SDF at *sdf_path*."""
    try:
        from ase.io import read
    except ImportError as exc:
        msg = "ASE is required: pip install ase"
        raise ImportError(msg) from exc

    traj = read(str(extxyz_path), index=":")
    n_frames = len(traj)
    print(f"[traj_to_sdf] Loaded {n_frames} frames from {extxyz_path}")

    # --- frame 0: build reference mol and perceive bonds ---
    ref_rwmol = atoms_to_rdmol(traj[0])
    rdDetermineBonds.DetermineBonds(ref_rwmol, charge=charge)
    ref_mol = ref_rwmol.GetMol()

    bond_types = {b.GetBondTypeAsDouble() for b in ref_mol.GetBonds()}
    print(
        f"[traj_to_sdf] Reference: {ref_mol.GetNumAtoms()} atoms, "
        f"{ref_mol.GetNumBonds()} bonds, types={sorted(bond_types)}"
    )

    # --- write all frames ---
    writer = Chem.SDWriter(str(sdf_path))
    writer.SetKekulize(False)  # preserve aromatic bond types

    for i, atoms in enumerate(traj):
        mol = update_conformer(ref_mol, atoms.positions)

        if i > 0:
            # Align this frame onto the reference (modifies conformer in place)
            rdMolAlign.AlignMol(mol, ref_mol)

        writer.write(mol)

    writer.close()
    print(f"[traj_to_sdf] Wrote {n_frames} aligned frames to {sdf_path}")


def main() -> None:
    if len(sys.argv) < 2:  # noqa: PLR2004
        print(__doc__)
        sys.exit(1)

    extxyz_path = Path(sys.argv[1])
    if len(sys.argv) >= 3:  # noqa: PLR2004
        sdf_path = Path(sys.argv[2])
    else:
        sdf_path = extxyz_path.with_suffix(".sdf")

    traj_to_sdf(extxyz_path, sdf_path)


if __name__ == "__main__":
    main()
