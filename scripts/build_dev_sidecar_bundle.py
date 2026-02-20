#!/usr/bin/env python3
"""
Build a minimal dev-mode python_runtime bundle for testing the download/install flow.
Points to the local dev Python environment and sidecar source.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a dev-mode python_runtime bundle")
    parser.add_argument("--sidecar-src", required=True, help="Path to llm-course-sidecar/ directory")
    parser.add_argument("--python-path", default=sys.executable, help="Path to Python executable")
    parser.add_argument("--version", default="0.1.0", help="Bundle version (semver)")
    parser.add_argument("--platform", default="dev-local", help="Platform scope_id (e.g. dev-local, py312-darwin-arm64)")
    parser.add_argument("--output", default="/tmp", help="Output directory")
    args = parser.parse_args()

    sidecar_src = Path(args.sidecar_src).resolve()
    python_path = Path(args.python_path).resolve()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not python_path.exists():
        print(f"ERROR: Python executable not found: {python_path}", file=sys.stderr)
        return 1

    out_path = output_dir / f"sidecar_bundle_{args.version}.tar.gz"

    # runtime.manifest.json tells the desktop where to find python and sidecar root
    manifest = {
        "format_version": "bundle-v1",
        "bundle_type": "python_runtime",
        "scope_id": args.platform,
        "version": args.version,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": {
            "executable_relpath": "start.sh",
        },
        "sidecar": {
            "root_relpath": "sidecar_root",
        },
    }

    with tarfile.open(out_path, "w:gz") as tf:
        def _add(name: str, data: bytes, mode: int = 0o644) -> None:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = mode
            tf.addfile(ti, io.BytesIO(data))

        # runtime.manifest.json
        _add("runtime.manifest.json", json.dumps(manifest, indent=2).encode())

        # start.sh — thin wrapper that calls the real Python
        start_sh = f"#!/bin/sh\nexec '{python_path}' \"$@\"\n".encode()
        _add("start.sh", start_sh, mode=0o755)

        # sidecar_root/app/server/main.py — entry point the desktop looks for
        # It imports and runs the installed sidecar package
        sidecar_main = (
            "import sys, os\n"
            f"sys.path.insert(0, '{sidecar_src / 'src'}')\n"
            "from sidecar.main import app\n"
            "import uvicorn\n"
            "if __name__ == '__main__':\n"
            "    uvicorn.run(app, host='127.0.0.1', port=8000)\n"
        ).encode()
        _add("sidecar_root/app/__init__.py", b"")
        _add("sidecar_root/app/server/__init__.py", b"")
        _add("sidecar_root/app/server/main.py", sidecar_main)

    sha256 = _sha256_bytes(out_path.read_bytes())
    size = out_path.stat().st_size
    print(f"Wrote:  {out_path}")
    print(f"SHA256: {sha256}")
    print(f"Size:   {size} bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
