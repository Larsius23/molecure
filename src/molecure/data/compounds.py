"""In-memory compounds, one class per dataset role.

The classes differ on purpose. A ScreeningSet has no `activities`, so no later
code can accidentally train on, or score against, screening data.
"""

from dataclasses import dataclass
from typing import ClassVar

from rdkit import Chem


# eq=False: two compounds are equal only if they are the same object. The default
# would compare every field, including the RDKit molecule, which is meaningless.
@dataclass(frozen=True, eq=False)
class Compound:
    id: str
    mol: Chem.Mol  # hydrogens exactly as in the source file
    origin: str  # where it came from, e.g. "calibration.csv line 7"


@dataclass(frozen=True, eq=False)
class LabeledCompound(Compound):
    activity: float  # on the p-scale (pIC50, pKi, ...)


@dataclass(frozen=True, eq=False)
class CalibrationSet:
    id: str
    endpoint: str
    compounds: tuple[LabeledCompound, ...]
    role: ClassVar[str] = "calibration"

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.compounds)

    @property
    def activities(self) -> tuple[float, ...]:
        return tuple(c.activity for c in self.compounds)

    def summary(self) -> dict:
        acts = self.activities
        return {
            "role": self.role,
            "n_compounds": len(self.compounds),
            "endpoint": self.endpoint,
            "activity_min": round(min(acts), 3),
            "activity_max": round(max(acts), 3),
            "activity_range": round(max(acts) - min(acts), 3),
        }


@dataclass(frozen=True, eq=False)
class ScreeningSet:
    id: str
    compounds: tuple[Compound, ...]
    role: ClassVar[str] = "screening"

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(c.id for c in self.compounds)

    def summary(self) -> dict:
        return {"role": self.role, "n_compounds": len(self.compounds)}


@dataclass(frozen=True, eq=False)
class ReferenceSet:
    id: str
    compound: Compound  # exactly one molecule, with a 3D pose
    role: ClassVar[str] = "reference"

    def summary(self) -> dict:
        mol = self.compound.mol
        return {
            "role": self.role,
            "n_compounds": 1,
            "n_atoms": mol.GetNumAtoms(),
            "n_heavy_atoms": mol.GetNumHeavyAtoms(),
            "has_explicit_h": any(a.GetAtomicNum() == 1 for a in mol.GetAtoms()),
        }


LoadedDataset = CalibrationSet | ScreeningSet | ReferenceSet
