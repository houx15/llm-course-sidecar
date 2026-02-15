#!/usr/bin/env python3
"""Restore chapter prompt/config files in monorepo content/curriculum from demo sources."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable, Optional, Sequence


PROMPT_FILES = ("interaction_protocol.md", "socratic_vs_direct.md")
CONSULTATION_FILES = ("consultation_config.yaml", "consultation_guide.md", "consultation_guide.json")


def _iter_course_chapters(content_curriculum: Path) -> Iterable[tuple[str, Path]]:
    courses_root = content_curriculum / "courses"
    if not courses_root.is_dir():
        return

    for course_dir in sorted(p for p in courses_root.iterdir() if p.is_dir()):
        chapters_dir = course_dir / "chapters"
        if not chapters_dir.is_dir():
            continue
        for chapter_dir in sorted(p for p in chapters_dir.iterdir() if p.is_dir()):
            yield course_dir.name, chapter_dir


def _copy_if_exists(source: Path, target: Path, dry_run: bool) -> bool:
    if not source.is_file():
        return False
    if dry_run:
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def restore_content_chapters(
    content_curriculum: Path,
    demo_curriculum: Path,
    companion_prompts_dir: Path,
    dry_run: bool = False,
) -> dict[str, int]:
    stats = {
        "chapters_seen": 0,
        "prompt_files_copied": 0,
        "consultation_files_copied": 0,
        "prompt_files_missing": 0,
        "chapters_missing_demo_match": 0,
    }

    for course_id, content_chapter_dir in _iter_course_chapters(content_curriculum):
        stats["chapters_seen"] += 1
        chapter_code = content_chapter_dir.name
        demo_chapter_dir = demo_curriculum / "courses" / course_id / "chapters" / chapter_code
        if not demo_chapter_dir.is_dir():
            stats["chapters_missing_demo_match"] += 1

        for filename in PROMPT_FILES:
            preferred = demo_chapter_dir / filename
            fallback = companion_prompts_dir / filename
            target = content_chapter_dir / filename
            if _copy_if_exists(preferred, target, dry_run):
                stats["prompt_files_copied"] += 1
                continue
            if _copy_if_exists(fallback, target, dry_run):
                stats["prompt_files_copied"] += 1
                continue
            stats["prompt_files_missing"] += 1

        for filename in CONSULTATION_FILES:
            source = demo_chapter_dir / filename
            if _copy_if_exists(source, content_chapter_dir / filename, dry_run):
                stats["consultation_files_copied"] += 1

    return stats


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Restore chapter interaction/socratic files and consultation assets from demo into content/curriculum"
    )
    parser.add_argument(
        "--content-curriculum",
        required=True,
        type=Path,
        help="Path to monorepo content/curriculum directory",
    )
    parser.add_argument(
        "--demo-curriculum",
        default=Path("demo/curriculum"),
        type=Path,
        help="Path to demo curriculum directory",
    )
    parser.add_argument(
        "--companion-prompts-dir",
        required=True,
        type=Path,
        help="Path to content/agents/companion directory for global prompt fallback files",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print summary only without copying files")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    stats = restore_content_chapters(
        content_curriculum=args.content_curriculum.resolve(),
        demo_curriculum=args.demo_curriculum.resolve(),
        companion_prompts_dir=args.companion_prompts_dir.resolve(),
        dry_run=args.dry_run,
    )

    mode = "DRY-RUN" if args.dry_run else "APPLY"
    print(f"[{mode}] chapters_seen={stats['chapters_seen']}")
    print(f"[{mode}] prompt_files_copied={stats['prompt_files_copied']}")
    print(f"[{mode}] consultation_files_copied={stats['consultation_files_copied']}")
    print(f"[{mode}] prompt_files_missing={stats['prompt_files_missing']}")
    print(f"[{mode}] chapters_missing_demo_match={stats['chapters_missing_demo_match']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
