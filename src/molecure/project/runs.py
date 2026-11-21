"""Run folders: one folder per execution, recording what produced the results.

runs/<timestamp>_<confighash>/
    config.resolved.yaml   the full config with every default written out
    manifest.json          versions, seed, input file fingerprints, dataset summaries
"""

import hashlib
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from itertools import count
from pathlib import Path

from rdkit import rdBase

import molecure
from molecure.config import ProjectConfig, dump_config

from .workspace import Workspace, WorkspaceError, referenced_files

TRACKED_PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "pydantic", "molscrub", "meeko")


def config_hash(cfg: ProjectConfig) -> str:
    """Fingerprint of the SETTINGS. Same settings -> same hash, however the YAML was written.

    Hashing the resolved config (not the YAML text) means comments, key order and
    omitted defaults make no difference.
    """
    canonical = json.dumps(cfg.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    """Fingerprint of a file's exact bytes. One changed character gives a different hash."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):  # 1 MiB at a time
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def environment_info() -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),  # "arm64" = native Apple Silicon
        "rdkit": rdBase.rdkitVersion,
        "packages": {name: _package_version(name) for name in TRACKED_PACKAGES},
    }


@dataclass(frozen=True)
class Run:
    root: Path
    run_id: str

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def read_manifest(self) -> dict:
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def write_manifest(self, manifest: dict) -> None:
        # Write to a temp file, then rename: a crash mid-write can't leave half a manifest.
        tmp = self.manifest_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.manifest_path)

    def add_to_manifest(self, key: str, value: object) -> None:
        manifest = self.read_manifest()
        manifest[key] = value
        self.write_manifest(manifest)


def create_run(ws: Workspace, cfg: ProjectConfig) -> Run:
    """Make a new run folder with the resolved config and a manifest."""
    missing = ws.missing_files(cfg)
    if missing:
        listing = ", ".join(f"{label} ({path})" for label, path in missing)
        raise WorkspaceError(f"cannot start a run; missing files: {listing}")

    digest = config_hash(cfg)
    now = datetime.now(UTC)
    base_id = f"{now:%Y%m%d-%H%M%S}_{digest[:8]}"
    runs_dir = ws.root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    # Two runs in the same second get -2, -3, ... mkdir() is the atomic "claim".
    for attempt in count(1):
        run_id = base_id if attempt == 1 else f"{base_id}-{attempt}"
        root = runs_dir / run_id
        try:
            root.mkdir()
            break
        except FileExistsError:
            continue

    dump_config(cfg, root / "config.resolved.yaml")

    inputs = []
    for label, rel in referenced_files(cfg):
        path = ws.root / rel
        inputs.append(
            {
                "label": label,
                "path": rel,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        )

    run = Run(root=root, run_id=run_id)
    run.write_manifest(
        {
            "run_id": run_id,
            "created_utc": now.isoformat(timespec="seconds"),
            "molecure_version": molecure.__version__,
            "config_hash": digest,
            "seed": cfg.runtime.seed,
            "environment": environment_info(),
            "inputs": inputs,
        }
    )
    return run
