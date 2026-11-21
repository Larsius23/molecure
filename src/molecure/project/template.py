"""The starter project.yaml written by `molecure init`."""

# __NAME__ is swapped for the folder name. (We use .replace() instead of an
# f-string because YAML uses { } itself.)
STARTER_YAML = """\
# mole.cure project file. All paths are relative to this folder.
schema_version: 1

project:
  name: __NAME__
  description: ""

datasets:
  # Calibration = structures + measured activity. Used for splitting, CV, fitting.
  - id: calibration_set
    role: calibration
    file: data/curated/calibration.csv
    id_col: compound_id
    smiles_col: smiles
    activity_col: pIC50
    endpoint: pIC50

  # Reference = one trusted 3D pose, the alignment template.
  - id: reference_ligand
    role: reference
    file: data/reference/reference.sdf

  # Screening = structures only. Prediction only; never touches the fit.
  # - id: screening_set
  #   role: screening
  #   file: data/screening/screening.csv

alignment:
  mode: mcs                  # mcs | o3a | prealigned (see schema for each mode's options)
  reference: reference_ligand
  min_core_atoms: 8

molecular_fields:
  method: comfa              # comfa | comsia
  grid:
    spacing: 2.0
    padding: 4.0

runtime:
  seed: 42
"""
