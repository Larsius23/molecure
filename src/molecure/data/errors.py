"""Errors raised while reading data files."""

MAX_SHOWN = 10


class DataError(Exception):
    """A data file cannot be used. Carries EVERY problem found, not just the first."""

    def __init__(self, source: str, problems: list[str]):
        self.source = source
        self.problems = list(problems)
        shown = self.problems[:MAX_SHOWN]
        lines = [f"{source}:"] + [f"  - {p}" for p in shown]
        hidden = len(self.problems) - len(shown)
        if hidden > 0:
            lines.append(f"  ... and {hidden} more")
        super().__init__("\n".join(lines))
