# Benchmark 0: the CoMFA steroid set

Purpose: check the 3D-QSAR engine against a published, well-studied dataset before using it on new data.

## Sources
- Cramer, Patterson & Bunce, J. Am. Chem. Soc. 1988, 110, 5959-5967: the original CoMFA analysis.
- Coats, Perspect. Drug Discov. Des. 1998, 12/13/14, 199-213: review of the dataset (raw affinity
  constants, corrected values, structural errors, CoMFA recomputed with leave-one-out).
- Structures: `source/Coats_CoMFA_steroidsBenchmarkDataset.sdf` (32 records, MOE 3D, explicit
  hydrogens, pre-aligned). The file name suggests Coats, but the file contains problems the chapter
  warns about (below), so it must not be assumed to be the corrected set.
  TODO: confirm provenance and redistribution terms before this repository goes public.

## Data
- Training: 21 steroids with CBG and TeBG affinity constants K (M^-1). Activity = log K = pKd.
- External: 10 steroids, CBG only. Not measured in the same assay: converted from free energies of
  binding in another study by a correlation (Coats, Eq. 1).
- Censored values: for 7 of the 21 CBG steroids no binding could be measured (K < 0.1e6). They are
  listed as log K = 5, so the real value is below 5. Kept for parity with the papers. The ChEMBL
  pipeline will NOT do this (relation "=" only).
- Steroid 31 (the only 9a-fluoro) is an outlier for every published method. Judge the external set
  with and without it (T10).

## What prepare.py changes
1. Drops source record 23: labelled cortisol 21-acetate but a ring-opened carboxylate anion.
2. Negates the activities (the source stores log 1/K; larger log K = stronger binding).
3. Corrects CBG of S16 and S17 from 5.225 to 5.255 (K = 0.18e6; Coats Table 1). The original value
   is kept in the `source_activity` tag.
4. Assigns ids from the paper's numbering (S01-S21 training, T01-T10 external).

## Known problem NOT fixed: S02 androstanediol
The 3-OH is alpha (should be beta, as Coats notes for the SYBYL set) and the 17-OH is alpha with
C17 undefined in the SMILES tag, so the 3D builder chose arbitrarily. Both TeBG and CBG results
depend on it. Default workspaces keep it ("as distributed", comparable to Coats's 0.665 row);
`prepare.py --exclude S02` leaves it out. Plan: rebuild it correctly once the aligner exists.

## Settings and deviations
- Generated projects use Coats's settings (LOO, scaling "block", sigma filter 2.0 kcal/mol).
- The "1/r dielectric" is read as a distance-dependent dielectric (energy ~ q*Q/r^2).
- We deviate on purpose: UFF Lennard-Jones parameters instead of Tripos, a probe radius near 1.9 A
  instead of 1.52 A, and truncated electrostatics are clipped instead of replaced by the mean of
  the other molecules.