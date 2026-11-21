"""Shared base class for every config section."""

from pydantic import BaseModel, ConfigDict

# Names for projects and datasets: letters, digits, underscore, hyphen.
# Strict on purpose: these end up in folder names and file names.
NAME_PATTERN = r"^[A-Za-z0-9_\-]+$"


class StrictModel(BaseModel):
    """Every config section inherits from this.

    extra="forbid": an unknown key (e.g. `phh: 7.4`) is an error, not silently ignored.
    frozen=True:    a loaded config cannot be modified, so settings can't drift mid-run.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
