#!/usr/bin/env python3
"""Build a chapter bundle tarball with a v2 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


REQUIRED_PROMPTS = (
    "chapter_context.md",
    "task_list.md",
    "task_completion_principles.md",
)

OPTIONAL_PROMPTS = (
    "interaction_protocol.md",
    "socratic_vs_direct.md",
    "consultation_config.yaml",
    "consultation_guide.md",
    "consultation_guide.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_prompts_dir(chapter_dir: Path) -> Path:
    prompts_dir = chapter_dir / "prompts"
    if prompts_dir.is_dir():
        source = prompts_dir
    else:
        source = chapter_dir

    missing = [name for name in REQUIRED_PROMPTS if not (source / name).is_file()]
    if missing:
        raise ValueError(
            "Missing required prompt files in chapter source: "
            + ", ".join(sorted(missing))
        )
    return source


def _copy_prompt_files(source_prompts_dir: Path, stage_prompts_dir: Path) -> List[Path]:
    stage_prompts_dir.mkdir(parents=True, exist_ok=True)
    copied: List[Path] = []
    for name in REQUIRED_PROMPTS + OPTIONAL_PROMPTS:
        src = source_prompts_dir / name
        if src.is_file():
            dest = stage_prompts_dir / name
            dest.write_bytes(src.read_bytes())
            copied.append(dest)
    return copied


def _copy_optional_dir(src: Optional[Path], dest: Path) -> bool:
    if not src or not src.is_dir():
        return False
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        target = dest / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    return True


def _infer_ids(chapter_dir: Path) -> tuple[str, str]:
    resolved = chapter_dir.resolve()
    parts = resolved.parts
    course_id = ""
    chapter_code = resolved.name

    # Choose the last valid courses/<course_id>/chapters/<chapter_code> pair.
    # This avoids brittle matches when parent path segments also contain "courses"/"chapters".
    selected_pair: Optional[tuple[str, str]] = None
    for chapter_idx in range(len(parts) - 2, -1, -1):
        if parts[chapter_idx] != "chapters":
            continue
        chapter_candidate = parts[chapter_idx + 1]
        for course_idx in range(chapter_idx - 1, -1, -1):
            if parts[course_idx] != "courses":
                continue
            if course_idx + 1 >= chapter_idx:
                continue
            selected_pair = (parts[course_idx + 1], chapter_candidate)
            break
        if selected_pair:
            break

    if selected_pair:
        course_id, chapter_code = selected_pair
    elif resolved.parent.name:
        course_id = resolved.parent.name
    return course_id, chapter_code


def _extract_title(prompt_source: Path, chapter_code: str) -> str:
    chapter_context = prompt_source / "chapter_context.md"
    if not chapter_context.is_file():
        return chapter_code
    for line in chapter_context.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip() or chapter_code
    return chapter_code


def _infer_required_experts(prompt_source: Path) -> List[str]:
    config_path = prompt_source / "consultation_config.yaml"
    if not config_path.is_file():
        return []
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    experts = payload.get("available_experts")
    if not isinstance(experts, list):
        return []
    return [str(item).strip() for item in experts if str(item).strip()]


def _collect_manifest_files(stage_root: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for path in sorted(stage_root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(stage_root)
        rel_str = rel.as_posix()
        if rel_str == "bundle.manifest.json":
            continue
        entries.append(
            {
                "path": rel_str,
                "sha256": _sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    return entries


def _build_manifest(
    stage_root: Path,
    prompt_source: Path,
    chapter_dir: Path,
    version: str,
    scope_id: Optional[str],
    course_id: Optional[str],
    chapter_code: Optional[str],
    title: Optional[str],
    required_experts: Optional[List[str]],
) -> Dict[str, Any]:
    inferred_course_id, inferred_chapter_code = _infer_ids(chapter_dir)
    final_course_id = course_id or inferred_course_id
    final_chapter_code = chapter_code or inferred_chapter_code
    final_title = title or _extract_title(prompt_source, final_chapter_code)
    final_scope_id = scope_id or f"{final_course_id}/{final_chapter_code}".strip("/")

    scripts_dir = stage_root / "scripts"
    datasets_dir = stage_root / "datasets"
    final_required_experts = (
        required_experts
        if required_experts is not None
        else _infer_required_experts(prompt_source)
    )

    return {
        "format_version": "bundle-v2",
        "bundle_type": "chapter",
        "scope_id": final_scope_id,
        "version": version,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "chapter": {
            "course_id": final_course_id,
            "chapter_code": final_chapter_code,
            "title": final_title,
            "has_scripts": scripts_dir.is_dir(),
            "has_datasets": datasets_dir.is_dir(),
            "required_experts": final_required_experts or [],
        },
        "files": _collect_manifest_files(stage_root),
    }


def build_chapter_bundle(
    chapter_dir: Path,
    output_dir: Path,
    version: str = "1.0.0",
    scope_id: Optional[str] = None,
    course_id: Optional[str] = None,
    chapter_code: Optional[str] = None,
    title: Optional[str] = None,
    scripts_dir: Optional[Path] = None,
    datasets_dir: Optional[Path] = None,
    assets_dir: Optional[Path] = None,
    required_experts: Optional[List[str]] = None,
    bundle_name: str = "chapter_bundle.tar.gz",
) -> tuple[Path, Dict[str, Any]]:
    chapter_dir = chapter_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_source = _resolve_prompts_dir(chapter_dir)
    scripts_source = scripts_dir.resolve() if scripts_dir else chapter_dir / "scripts"
    datasets_source = datasets_dir.resolve() if datasets_dir else chapter_dir / "datasets"
    assets_source = assets_dir.resolve() if assets_dir else chapter_dir / "assets"

    with tempfile.TemporaryDirectory(prefix="chapter-bundle-") as tmpdir:
        stage_root = Path(tmpdir)
        stage_prompts = stage_root / "prompts"
        _copy_prompt_files(prompt_source, stage_prompts)
        _copy_optional_dir(scripts_source, stage_root / "scripts")
        _copy_optional_dir(datasets_source, stage_root / "datasets")
        _copy_optional_dir(assets_source, stage_root / "assets")

        manifest = _build_manifest(
            stage_root=stage_root,
            prompt_source=prompt_source,
            chapter_dir=chapter_dir,
            version=version,
            scope_id=scope_id,
            course_id=course_id,
            chapter_code=chapter_code,
            title=title,
            required_experts=required_experts,
        )

        manifest_path = stage_root / "bundle.manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        output_path = output_dir / bundle_name
        with tarfile.open(output_path, "w:gz") as archive:
            for path in sorted(stage_root.rglob("*")):
                if path.is_file():
                    archive.add(path, arcname=path.relative_to(stage_root).as_posix(), recursive=False)

    return output_path, manifest


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build chapter bundle tar.gz with manifest")
    parser.add_argument("--chapter-dir", required=True, type=Path, help="Chapter directory path")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--version", default="1.0.0", help="Bundle semantic version")
    parser.add_argument("--scope-id", default=None, help="Override manifest scope_id")
    parser.add_argument("--course-id", default=None, help="Override chapter.course_id")
    parser.add_argument("--chapter-code", default=None, help="Override chapter.chapter_code")
    parser.add_argument("--title", default=None, help="Override chapter title")
    parser.add_argument("--scripts-dir", default=None, type=Path, help="Optional scripts source override")
    parser.add_argument("--datasets-dir", default=None, type=Path, help="Optional datasets source override")
    parser.add_argument("--assets-dir", default=None, type=Path, help="Optional assets source override")
    parser.add_argument(
        "--required-expert",
        action="append",
        default=None,
        help="Append required expert id (repeatable)",
    )
    parser.add_argument("--bundle-name", default="chapter_bundle.tar.gz", help="Output tar.gz file name")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    output_path, manifest = build_chapter_bundle(
        chapter_dir=args.chapter_dir,
        output_dir=args.output,
        version=args.version,
        scope_id=args.scope_id,
        course_id=args.course_id,
        chapter_code=args.chapter_code,
        title=args.title,
        scripts_dir=args.scripts_dir,
        datasets_dir=args.datasets_dir,
        assets_dir=args.assets_dir,
        required_experts=args.required_expert,
        bundle_name=args.bundle_name,
    )
    print(f"bundle: {output_path}")
    print(f"scope_id: {manifest['scope_id']}")
    print(f"files: {len(manifest['files'])}")
    print(f"has_scripts: {manifest['chapter']['has_scripts']}")
    print(f"has_datasets: {manifest['chapter']['has_datasets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
