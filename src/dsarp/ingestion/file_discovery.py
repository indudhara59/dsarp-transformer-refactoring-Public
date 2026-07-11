"""Configurable Arcan input discovery."""
from pathlib import Path

from dsarp.constants import INPUT_FILENAMES
from dsarp.exceptions import InputFileError


def discover_files(data_dir: Path, filenames: dict[str, str] | None = None) -> dict[str, Path]:
    """Find each required export below data_dir, rejecting ambiguity."""
    names = filenames or INPUT_FILENAMES
    if not data_dir.exists():
        raise InputFileError(f"Data directory does not exist: {data_dir}")
    result: dict[str, Path] = {}
    for role, filename in names.items():
        matches = sorted(p for p in data_dir.rglob(filename) if p.is_file())
        if not matches:
            raise InputFileError(f"Missing required {role} file: {filename} below {data_dir}")
        if len(matches) > 1:
            direct = [p for p in matches if p.parent in {data_dir, data_dir / "raw"}]
            if len(direct) != 1:
                raise InputFileError(f"Ambiguous {role} file; found: {matches}")
            matches = direct
        result[role] = matches[0]
    return result

