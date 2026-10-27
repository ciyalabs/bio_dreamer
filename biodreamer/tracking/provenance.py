"""captures the context a run happened in, without being asked.

this is the part that stops results getting mixed up later: a metric is only
interpretable alongside the code, config, data and seed that produced it. every
field here is collected automatically when a run opens.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Provenance:
    git_sha: str | None
    git_branch: str | None
    git_dirty: bool
    host: str
    user: str
    python: str
    platform: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def collect() -> Provenance:
    """gathers code and environment provenance. never raises."""
    sha = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    return Provenance(
        git_sha=sha,
        git_branch=branch,
        # a None status means git failed; treat that as dirty rather than clean,
        # so an unverifiable run is never silently reported as reproducible.
        git_dirty=True if status is None else bool(status),
        host=socket.gethostname(),
        user=os.environ.get("USER") or os.environ.get("USERNAME") or "unknown",
        python=sys.version.split()[0],
        platform=platform.platform(),
    )


def hash_object(obj: Any) -> str:
    """stable short hash of any json-serialisable object.

    used for config and variant identity, so two runs with the same settings
    collide on purpose and can be detected as repeats.
    """
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def hash_paths(paths: list[str | Path]) -> str | None:
    """hashes the contents of dataset/manifest files, for data identity.

    directories are hashed over their file names and sizes rather than contents,
    which is cheap and enough to catch a swapped dataset. returns None if nothing
    resolvable was given.
    """
    digest = hashlib.sha256()
    found = False
    for p in sorted(str(x) for x in paths):
        path = Path(p)
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
            found = True
        elif path.is_dir():
            for f in sorted(path.rglob("*")):
                if f.is_file():
                    digest.update(str(f.relative_to(path)).encode())
                    digest.update(str(f.stat().st_size).encode())
                    found = True
    return digest.hexdigest()[:12] if found else None


def variant_key(variant: dict[str, Any]) -> str:
    """canonical identity of a variant, independent of key order."""
    return hash_object(variant or {})


def variant_label(variant: dict[str, Any]) -> str:
    """short human-readable variant label, e.g. 'landscape=GB1-4site, seed=1'."""
    if not variant:
        return "(default)"
    return ", ".join(f"{k}={variant[k]}" for k in sorted(variant))
