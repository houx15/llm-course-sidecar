#!/usr/bin/env python3
"""
Build a platform-agnostic sidecar code bundle (bundle-v2).

The bundle is self-contained: it includes the full sidecar source tree and a
requirements.txt derived from pyproject.toml.  The desktop will install deps
into a conda env and launch the entry point:

    app/server/main.py  →  adds src/ to sys.path, then imports sidecar.main:app

Usage:
    python scripts/build_sidecar_code_bundle.py --version 0.2.0 --output /tmp/
    python scripts/build_sidecar_code_bundle.py --version 0.2.0 --scope-id core --output /tmp/
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _add_bytes(tf: tarfile.TarFile, arcname: str, data: bytes, mode: int = 0o644) -> None:
    """Add raw bytes to a TarFile under the given archive name."""
    ti = tarfile.TarInfo(name=arcname)
    ti.size = len(data)
    ti.mode = mode
    tf.addfile(ti, io.BytesIO(data))


def _add_tree(tf: tarfile.TarFile, src_dir: Path, arcname_prefix: str) -> None:
    """
    Recursively add all files under src_dir into the archive, placing them
    under arcname_prefix/.  Skips __pycache__ directories and .pyc files.
    """
    for path in sorted(src_dir.rglob("*")):
        # Skip __pycache__ directories and compiled bytecode
        if "__pycache__" in path.parts:
            continue
        if path.suffix == ".pyc":
            continue
        if not path.is_file():
            continue

        rel = path.relative_to(src_dir)
        arcname = f"{arcname_prefix}/{rel}"
        tf.add(str(path), arcname=arcname, recursive=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a platform-agnostic sidecar code bundle (bundle-v2)"
    )
    parser.add_argument("--version", required=True, help="Bundle version (semver), e.g. 0.2.0")
    parser.add_argument("--scope-id", default="core", help="Logical scope identifier (default: core)")
    parser.add_argument("--output", default="/tmp", help="Output directory (default: /tmp)")
    args = parser.parse_args()

    # Resolve repo root relative to this script (scripts/ → repo root)
    repo_root = Path(__file__).resolve().parent.parent
    pyproject_path = repo_root / "pyproject.toml"
    src_dir = repo_root / "src" / "sidecar"

    # Validate inputs
    if not pyproject_path.exists():
        print(f"ERROR: pyproject.toml not found at {pyproject_path}", file=sys.stderr)
        return 1
    if not src_dir.exists():
        print(f"ERROR: sidecar source dir not found at {src_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_filename = f"sidecar_code_{args.version}.tar.gz"
    out_path = output_dir / out_filename

    # ------------------------------------------------------------------
    # Parse requirements from pyproject.toml [project.dependencies]
    # ------------------------------------------------------------------
    with open(pyproject_path, "rb") as fh:
        pyproject = tomllib.load(fh)

    deps: list[str] = pyproject.get("project", {}).get("dependencies", [])
    requirements_txt = "\n".join(deps) + "\n"

    # ------------------------------------------------------------------
    # Build runtime.manifest.json  (bundle-v2: no python.executable_relpath)
    # ------------------------------------------------------------------
    manifest = {
        "format_version": "bundle-v2",
        "bundle_type": "python_runtime",
        "scope_id": args.scope_id,
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {},
        "sidecar": {
            "root_relpath": ".",
        },
    }

    # ------------------------------------------------------------------
    # app/server/main.py  — entry point with sys.path hack
    # ------------------------------------------------------------------
    # Path arithmetic (from app/server/main.py → src/):
    #   __file__ is   <bundle>/app/server/main.py
    #   dirname x2 →  <bundle>/
    #   + 'src'     →  <bundle>/src
    app_server_main = (
        "import sys, os\n"
        "_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'src')\n"
        "sys.path.insert(0, _src)\n"
        "from sidecar.main import app\n"
    )

    # ------------------------------------------------------------------
    # Assemble the tar.gz in memory then write once
    # ------------------------------------------------------------------
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        # runtime.manifest.json
        _add_bytes(tf, "runtime.manifest.json", json.dumps(manifest, indent=2).encode())

        # requirements.txt
        _add_bytes(tf, "requirements.txt", requirements_txt.encode())

        # app/__init__.py
        _add_bytes(tf, "app/__init__.py", b"")

        # app/server/__init__.py
        _add_bytes(tf, "app/server/__init__.py", b"")

        # app/server/main.py
        _add_bytes(tf, "app/server/main.py", app_server_main.encode())

        # src/sidecar/** (full source tree, no __pycache__, no .pyc)
        _add_tree(tf, src_dir, arcname_prefix="src/sidecar")

    bundle_bytes = buf.getvalue()
    out_path.write_bytes(bundle_bytes)

    sha256 = _sha256_bytes(bundle_bytes)
    size = len(bundle_bytes)
    size_mb = size / (1024 * 1024)

    print(f"Wrote:  {out_path}")
    print(f"SHA256: {sha256}")
    print(f"Size:   {size} bytes ({size_mb:.2f} MB)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
