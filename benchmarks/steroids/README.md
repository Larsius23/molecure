# Benchmark 0: the CoMFA steroid set

Purpose: check the 3D-QSAR engine against a published, well-studied dataset before using it on new data.

## Sources
- Method and reference values: Cramer, Patterson & Bunce, J. Am. Chem. Soc. 1988, 110, 5959-5967.
- Structures: `source/Coats_CoMFA_steroidsBenchmarkDataset.sdf` (32 records, MOE 3D, explicit
  hydrogens, pre-aligned). I could not verify the file's origin from its contents.
  TODO: confirm provenance and redistribution terms before this repository goes public.

## Data
- 21 training steroids (Cramer Table I) with binding to CBG and TeBG (called TBG in the paper's text).
- 10 external steroids (Cramer Figure 3), CBG only. Per the paper these values were converted from
  another dataset's free energies, not measured in the same assay.

## What prepare.py changes
1. Drops source record 23: it is labelled cortisol 21-acetate but is a ring-opened carboxylate anion.
2. Negates activities: the source stores log-K values where more negative = stronger binding.
3. Assigns ids from the paper's numbering (S01-S21 training, T01-T10 external).

## Known issues
- 7 of 21 CBG training values are exactly 5.000 (detection-limit placeholders, probably).
- S16 etiocholanolone and S17 pregnenolone: CBG is 5.225 in the SDF but 5.255 in the paper's Table I.
  Unresolved; the SDF values are used. A test pins this difference so it cannot change unnoticed.
- The paper's "1/r dielectric" is read as a distance-dependent dielectric (energy ~ q*Q/r^2).
  Benchmark 0 can test this reading against a constant dielectric.
- We deviate from the paper on purpose: UFF Lennard-Jones parameters instead of Tripos, a probe
  radius near 1.9 A instead of 1.52 A, and truncated electrostatics are clipped, not replaced by the
  mean of the other molecules.