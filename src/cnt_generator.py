"""Utilities for generating carbon nanotube geometries.

This module provides helper routines for constructing armchair and zigzag
carbon nanotube coordinates.  It uses ASE when available and falls back to a
minimal internal implementation when ASE cannot be imported.  The fallback
follows the standard chiral vector construction to wrap a graphene sheet onto
an ideal cylinder.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    from ase import Atoms  # type: ignore
    from ase.build import nanotube  # type: ignore
except Exception:  # pragma: no cover - ASE is optional in the execution env
    Atoms = None  # type: ignore
    nanotube = None  # type: ignore


@dataclass(frozen=True)
class CNTGeometry:
    """Container describing a carbon nanotube geometry.

    Attributes:
        n: First chiral index.
        m: Second chiral index.
        bond_length: Carbon--carbon bond length in Å.
        chiral_vector: Vector along the circumference (Å).
        translation_vector: Primitive vector along the nanotube axis (Å).
        circumference: Circumference magnitude (Å).
        period: Length of the translational unit cell (Å).
        num_subbands: Number of one-dimensional subbands (N).  The total number
            of atoms per translational cell is ``2 * num_subbands``.
        positions: Array with Cartesian coordinates (Å) for every atom in the
            generated geometry.  The tube axis points along the z direction in
            the fallback implementation.
        metadata: Extra information such as the transformation matrix between
            graphene lattice vectors and the nanotube basis.
        ase_atoms: ASE ``Atoms`` object when ASE is available, otherwise
            ``None``.
    """

    n: int
    m: int
    bond_length: float
    chiral_vector: np.ndarray
    translation_vector: np.ndarray
    circumference: float
    period: float
    num_subbands: int
    positions: np.ndarray
    metadata: Dict[str, np.ndarray]
    ase_atoms: Optional["Atoms"] = None


# Graphene lattice and nearest-neighbour vectors used in the fallback builder.
SQRT3 = math.sqrt(3.0)
A1_BASE = np.array([SQRT3 / 2.0, 3.0 / 2.0, 0.0])
A2_BASE = np.array([-SQRT3 / 2.0, 3.0 / 2.0, 0.0])
DELTA_BASE = (
    np.array([0.0, 1.0, 0.0]),
    np.array([SQRT3 / 2.0, -0.5, 0.0]),
    np.array([-SQRT3 / 2.0, -0.5, 0.0]),
)


def _compute_lattice_vectors(bond_length: float) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, ...]]:
    a1 = bond_length * A1_BASE
    a2 = bond_length * A2_BASE
    deltas = tuple(bond_length * d for d in DELTA_BASE)
    return a1, a2, deltas


def _generate_cnt_positions(
    n: int,
    m: int,
    length_repeats: int,
    bond_length: float,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    a1, a2, _ = _compute_lattice_vectors(bond_length)
    chiral_vector = n * a1 + m * a2
    d_r = math.gcd(2 * m + n, 2 * n + m)
    t1 = (2 * m + n) // d_r
    t2 = -(2 * n + m) // d_r
    translation_vector = t1 * a1 + t2 * a2

    circumference = np.linalg.norm(chiral_vector)
    period = np.linalg.norm(translation_vector)
    c_hat = chiral_vector[:2] / circumference
    t_hat = translation_vector[:2] / period

    # Transformation matrix to express graphene coordinates in the CNT basis.
    transform = np.column_stack((chiral_vector[:2], translation_vector[:2]))
    transform_inv = np.linalg.inv(transform)

    # Determine a sampling window large enough to populate the nanotube cell.
    search_extent = max(abs(n), abs(m), abs(t1), abs(t2)) + 2 * length_repeats + 4
    tau_a = np.zeros(3)
    tau_b = np.array([0.0, bond_length, 0.0])
    candidate_vectors: List[np.ndarray] = []
    for i in range(-search_extent, search_extent + 1):
        for j in range(-search_extent, search_extent + 1):
            lattice_shift = i * a1 + j * a2
            for tau in (tau_a, tau_b):
                pos2d = lattice_shift + tau
                uv = transform_inv @ pos2d[:2]
                u, v = uv
                if (-1e-8 <= u < 1.0 - 1e-8) and (-1e-8 <= v < length_repeats - 1e-8):
                    candidate_vectors.append(np.array([u, v, tau is tau_b], dtype=float))

    if not candidate_vectors:
        raise RuntimeError("Failed to generate CNT lattice vectors: search window too small.")

    # Remove duplicates and sort.
    unique: Dict[Tuple[int, int, int], Tuple[float, float, float]] = {}
    for u, v, sublattice in candidate_vectors:
        key = (int(round(u * 1e6)), int(round(v * 1e6)), int(sublattice))
        if key not in unique:
            unique[key] = (u, v, sublattice)
    uv_list = sorted(unique.values(), key=lambda item: (item[1], item[0], item[2]))

    radius = circumference / (2.0 * math.pi)
    positions = []
    for u, v, _ in uv_list:
        angle = 2.0 * math.pi * u
        z = v * period
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        positions.append([x, y, z])

    metadata = {
        "a1": a1,
        "a2": a2,
        "transform": transform,
        "transform_inv": transform_inv,
        "chiral_hat": c_hat,
        "translation_hat": t_hat,
    }
    return np.array(positions), metadata


def generate_cnt(
    n: int,
    m: int,
    *,
    length_repeats: int = 4,
    bond_length: float = 1.42,
    use_ase: bool = True,
) -> CNTGeometry:
    """Generate a carbon nanotube geometry.

    Args:
        n: First chiral index.
        m: Second chiral index.
        length_repeats: Number of translational unit cells along the nanotube
            axis.  Increasing this value increases the total length of the
            generated structure.
        bond_length: Carbon--carbon bond length in Å.
        use_ase: When ``True`` (default) the function attempts to use the ASE
            nanotube generator.  If ASE is not available the function falls back
            to an internal implementation.

    Returns:
        A :class:`CNTGeometry` instance containing both the Cartesian positions
        and auxiliary information required for band-structure calculations.
    """

    if use_ase and nanotube is not None and Atoms is not None:  # pragma: no branch
        atoms = nanotube(n, m, length_repeats, bond=bond_length)
        chiral_vector = atoms.get_cell()[0]
        translation_vector = atoms.get_cell()[2]
        circumference = np.linalg.norm(chiral_vector)
        period = np.linalg.norm(translation_vector)
        d_r = math.gcd(2 * m + n, 2 * n + m)
        num_subbands = 2 * (n * n + m * m + n * m) // d_r
        positions = atoms.get_positions()
        metadata = {
            "a1": atoms.cell[0],
            "a2": atoms.cell[1],
            "chiral_hat": chiral_vector[:2] / circumference,
            "translation_hat": translation_vector[:2] / period if period else translation_vector[:2],
        }
        return CNTGeometry(
            n=n,
            m=m,
            bond_length=bond_length,
            chiral_vector=np.asarray(chiral_vector),
            translation_vector=np.asarray(translation_vector),
            circumference=float(circumference),
            period=float(period),
            num_subbands=int(num_subbands),
            positions=np.asarray(positions),
            metadata=metadata,
            ase_atoms=atoms,
        )

    # Fallback pathway using an internal graphene wrapping routine.
    positions, metadata = _generate_cnt_positions(n, m, length_repeats, bond_length)
    chiral_vector = n * metadata["a1"] + m * metadata["a2"]
    d_r = math.gcd(2 * m + n, 2 * n + m)
    circumference = np.linalg.norm(chiral_vector)
    period = float(np.linalg.norm(metadata["transform"][:, 1]))
    translation_vector = np.array([0.0, 0.0, period])
    num_subbands = 2 * (n * n + m * m + n * m) // d_r
    return CNTGeometry(
        n=n,
        m=m,
        bond_length=bond_length,
        chiral_vector=chiral_vector,
        translation_vector=translation_vector,
        circumference=float(circumference),
        period=float(period),
        num_subbands=int(num_subbands),
        positions=positions,
        metadata=metadata,
        ase_atoms=None,
    )


def armchair_cnt(length_repeats: int = 4, bond_length: float = 1.42) -> CNTGeometry:
    """Generate a representative metallic (5,5) armchair CNT."""

    return generate_cnt(5, 5, length_repeats=length_repeats, bond_length=bond_length)


def zigzag_cnt(length_repeats: int = 4, bond_length: float = 1.42) -> CNTGeometry:
    """Generate a representative (9,0) zig-zag CNT."""

    return generate_cnt(9, 0, length_repeats=length_repeats, bond_length=bond_length)


def graphene_nearest_neighbor_vectors(bond_length: float = 1.42) -> Tuple[np.ndarray, ...]:
    """Return the three nearest-neighbour vectors of graphene."""

    _, _, deltas = _compute_lattice_vectors(bond_length)
    return deltas


def chiral_parameters(n: int, m: int, bond_length: float = 1.42) -> Dict[str, float]:
    """Return useful CNT parameters for tight-binding calculations."""

    geometry = generate_cnt(n, m, length_repeats=1, bond_length=bond_length, use_ase=False)
    transform = geometry.metadata["transform"]
    reciprocal_basis = 2.0 * math.pi * geometry.metadata["transform_inv"].T
    return {
        "circumference": geometry.circumference,
        "period": geometry.period,
        "num_subbands": geometry.num_subbands,
        "chiral_hat": geometry.metadata["chiral_hat"],
        "translation_hat": geometry.metadata["translation_hat"],
        "reciprocal_basis": reciprocal_basis,
    }
