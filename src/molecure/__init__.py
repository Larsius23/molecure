"""mole.cure: target-agnostic 3D-QSAR for structure-based drug discovery."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("molecure")
except PackageNotFoundError:
    __version__ = "0+unknown"
