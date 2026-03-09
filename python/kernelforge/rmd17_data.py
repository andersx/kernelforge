"""
Loader for the Revised MD17 (rMD17) dataset in NPZ format.

The rMD17 dataset provides recalculated energies and forces at the PBE/def2-SVP
level of theory for 10 small organic molecules, with pre-made train/test splits
(1000 train + 1000 test frames per split).

Source: https://github.com/andersx/rmd17-npz

Citation:
    Anders S. Christensen and O. Anatole von Lilienfeld (2020)
    "On the role of gradients for machine learning of molecular energies and forces"
    https://arxiv.org/abs/2007.09593

NPZ key mapping (rMD17 → this module):
    nuclear_charges → z   (N_atoms,)         atomic numbers, uint8
    coords          → R   (N_frames, N_atoms, 3)  Cartesian coords, Å, float64
    energies        → E   (N_frames,)         total energy, kcal/mol, float64
    forces          → F   (N_frames, N_atoms, 3)  forces, kcal/mol/Å, float64

Note on force sign convention:
    rMD17 forces are F = -dE/dR (negative gradient), same convention as the
    kernelforge KRR examples which use forces directly as training labels.
    The local KRR example negates forces loaded from the original MD17 dataset
    (F = -data["F"]) because that dataset stores the gradient (+dE/dR) rather
    than the force.  rMD17 forces are already in the force convention and
    should NOT be negated.
"""

import sys
import urllib.request
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Cache directory (shared with cli.py)
# ---------------------------------------------------------------------------
_CACHE_DIR = Path.home() / ".kernelforge" / "datasets"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RMD17_BASE_URL = "https://raw.githubusercontent.com/andersx/rmd17-npz/master/rmd17-npz/"


def _download_rmd17_npz(filename: str) -> Path:
    """Download a single rMD17 NPZ file to the cache directory if not present.

    Parameters
    ----------
    filename:
        Bare filename, e.g. ``"rmd17_ethanol_train_01.npz"``.

    Returns
    -------
    Path
        Absolute path to the cached file.
    """
    path = _CACHE_DIR / filename
    if not path.exists():
        url = _RMD17_BASE_URL + filename
        print(f"  [Downloading {filename} from rmd17-npz...]")
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status != 200:
                    raise RuntimeError(f"HTTP {response.status}: failed to download {url}")
                path.write_bytes(response.read())
        except Exception as exc:
            print(f"  [Error downloading {filename}: {exc}]", file=sys.stderr)
            raise
    return path


def _load_rmd17_npz(path: Path) -> dict[str, Any]:
    """Load a single rMD17 NPZ file and normalise key names.

    Returns a dict with keys ``z``, ``R``, ``E``, ``F`` matching the
    convention used by ``kernelforge.cli.load_ethanol_raw_data``.
    """
    npz = np.load(path, allow_pickle=False)
    return {
        "z": npz["nuclear_charges"].astype(np.int32),  # (N_atoms,)
        "R": npz["coords"],  # (N_frames, N_atoms, 3)
        "E": npz["energies"],  # (N_frames,)
        "F": npz["forces"],  # (N_frames, N_atoms, 3)
    }


# ---------------------------------------------------------------------------
# Public loaders — one per molecule / split
# ---------------------------------------------------------------------------


@cache
def load_rmd17_ethanol_split01() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load rMD17 ethanol train/test split 01 (1000 + 1000 frames).

    Downloads both NPZ files from the andersx/rmd17-npz GitHub repository on
    first call and caches them in ``~/.kernelforge/datasets/``.  The loaded
    arrays are kept in memory so repeated calls within the same process are
    free.

    Returns
    -------
    train : dict
        Keys ``z``, ``R``, ``E``, ``F`` for the 1000 training frames.
    test : dict
        Keys ``z``, ``R``, ``E``, ``F`` for the 1000 test frames.

    Notes
    -----
    * ``z`` shape: ``(9,)``  — atomic numbers (H, C, O for ethanol)
    * ``R`` shape: ``(1000, 9, 3)``
    * ``E`` shape: ``(1000,)``  — energies in kcal/mol
    * ``F`` shape: ``(1000, 9, 3)``  — forces in kcal/mol/Å  (F = -dE/dR)
    """
    train_path = _download_rmd17_npz("rmd17_ethanol_train_01.npz")
    test_path = _download_rmd17_npz("rmd17_ethanol_test_01.npz")
    return _load_rmd17_npz(train_path), _load_rmd17_npz(test_path)
