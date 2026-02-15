import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

import sidecar.main as sidecar_main


def test_find_chapter_dir_in_bundle_supports_flat_layout(tmp_path):
    bundle_root = tmp_path / "bundle"
    chapter_dir = bundle_root / "content" / "curriculum" / "courses" / "course_1" / "chapters" / "ch_01"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    (chapter_dir / "chapter_context.md").write_text("# chapter", encoding="utf-8")
    (chapter_dir / "task_list.md").write_text("- task", encoding="utf-8")
    (chapter_dir / "task_completion_principles.md").write_text("- principle", encoding="utf-8")

    resolved = sidecar_main._find_chapter_dir_in_bundle(bundle_root, "course_1", "ch_01")
    assert resolved == chapter_dir


def test_find_chapter_dir_in_bundle_supports_prompts_subdir_with_resources(tmp_path):
    bundle_root = tmp_path / "bundle"
    chapter_root = bundle_root / "courses" / "course_1" / "chapters" / "ch_01"
    prompts_dir = chapter_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    (prompts_dir / "chapter_context.md").write_text("# chapter", encoding="utf-8")
    (prompts_dir / "task_list.md").write_text("- task", encoding="utf-8")
    (prompts_dir / "task_completion_principles.md").write_text("- principle", encoding="utf-8")
    (chapter_root / "scripts").mkdir(parents=True, exist_ok=True)
    (chapter_root / "datasets").mkdir(parents=True, exist_ok=True)

    resolved_prompts = sidecar_main._find_chapter_dir_in_bundle(bundle_root, "course_1", "ch_01")
    layout = sidecar_main._resolve_chapter_bundle_layout(bundle_root, "course_1", "ch_01")

    assert resolved_prompts == prompts_dir
    assert layout is not None
    assert layout.chapter_root == chapter_root
    assert layout.prompts_dir == prompts_dir
    assert layout.scripts_dir == chapter_root / "scripts"
    assert layout.datasets_dir == chapter_root / "datasets"


def test_resolve_curriculum_dir_copies_scripts_datasets_assets(tmp_path, monkeypatch):
    default_curriculum = tmp_path / "default_curriculum"
    (default_curriculum / "_templates").mkdir(parents=True, exist_ok=True)
    (default_curriculum / "_templates" / "dynamic_report_template.md").write_text("template", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    chapter_root = bundle_root / "courses" / "course_1" / "chapters" / "ch_01"
    prompts_dir = chapter_root / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    for name, content in [
        ("chapter_context.md", "# chapter"),
        ("task_list.md", "- task"),
        ("task_completion_principles.md", "- principle"),
        ("interaction_protocol.md", "- protocol"),
        ("socratic_vs_direct.md", "- mode"),
        ("consultation_guide.md", "guide"),
    ]:
        (prompts_dir / name).write_text(content, encoding="utf-8")

    scripts_dir = chapter_root / "scripts"
    datasets_dir = chapter_root / "datasets"
    assets_dir = chapter_root / "assets" / "images"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)
    assets_dir.mkdir(parents=True, exist_ok=True)
    (scripts_dir / "starter_code.py").write_text("print('hi')", encoding="utf-8")
    (datasets_dir / "sample.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    (assets_dir / "plot.txt").write_text("img", encoding="utf-8")

    monkeypatch.setattr(sidecar_main, "default_curriculum_dir", default_curriculum)
    monkeypatch.setattr(sidecar_main, "overlay_root", tmp_path / "sessions" / "_chapter_overlays")

    resolved = sidecar_main._resolve_curriculum_dir(
        "course_1/ch_01",
        {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
    )
    chapter_target = resolved / "courses" / "course_1" / "chapters" / "ch_01"

    assert (chapter_target / "chapter_context.md").exists()
    assert (chapter_target / "interaction_protocol.md").exists()
    assert (chapter_target / "consultation_guide.md").exists()
    assert (chapter_target / "scripts" / "starter_code.py").exists()
    assert (chapter_target / "datasets" / "sample.csv").exists()
    assert (chapter_target / "assets" / "images" / "plot.txt").exists()


def test_resolve_curriculum_dir_isolates_overlay_per_bundle_version(tmp_path, monkeypatch):
    default_curriculum = tmp_path / "default_curriculum"
    (default_curriculum / "_templates").mkdir(parents=True, exist_ok=True)
    (default_curriculum / "_templates" / "dynamic_report_template.md").write_text("template", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    chapter_root = bundle_root / "courses" / "course_1" / "chapters" / "ch_01"
    prompts_dir = chapter_root / "prompts"
    scripts_dir = chapter_root / "scripts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    for name, content in [
        ("chapter_context.md", "# chapter"),
        ("task_list.md", "- task"),
        ("task_completion_principles.md", "- principle"),
        ("consultation_guide.md", "old-guide"),
    ]:
        (prompts_dir / name).write_text(content, encoding="utf-8")
    (scripts_dir / "old_script.py").write_text("print('old')", encoding="utf-8")
    (bundle_root / "bundle.manifest.json").write_text('{"version":"1"}', encoding="utf-8")

    monkeypatch.setattr(sidecar_main, "default_curriculum_dir", default_curriculum)
    monkeypatch.setattr(sidecar_main, "overlay_root", tmp_path / "sessions" / "_chapter_overlays")

    resolved_v1 = sidecar_main._resolve_curriculum_dir(
        "course_1/ch_01",
        {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
    )
    chapter_v1 = resolved_v1 / "courses" / "course_1" / "chapters" / "ch_01"
    assert (chapter_v1 / "scripts" / "old_script.py").exists()
    assert (chapter_v1 / "consultation_guide.md").exists()

    (scripts_dir / "old_script.py").unlink()
    (prompts_dir / "consultation_guide.md").unlink()
    (scripts_dir / "new_script.py").write_text("print('new')", encoding="utf-8")
    time.sleep(0.001)
    (bundle_root / "bundle.manifest.json").write_text('{"version":"2"}', encoding="utf-8")

    resolved_v2 = sidecar_main._resolve_curriculum_dir(
        "course_1/ch_01",
        {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
    )
    chapter_v2 = resolved_v2 / "courses" / "course_1" / "chapters" / "ch_01"

    assert resolved_v2 != resolved_v1
    assert not (chapter_v2 / "scripts" / "old_script.py").exists()
    assert not (chapter_v2 / "consultation_guide.md").exists()
    assert (chapter_v2 / "scripts" / "new_script.py").exists()


def test_resolve_curriculum_dir_rejects_path_traversal_chapter_id(tmp_path, monkeypatch):
    default_curriculum = tmp_path / "default_curriculum"
    default_curriculum.mkdir(parents=True, exist_ok=True)

    bundle_root = tmp_path / "bundle"
    chapter_root = bundle_root / "courses" / "course_1" / "chapters" / "ch_01" / "prompts"
    chapter_root.mkdir(parents=True, exist_ok=True)
    for name, content in [
        ("chapter_context.md", "# chapter"),
        ("task_list.md", "- task"),
        ("task_completion_principles.md", "- principle"),
    ]:
        (chapter_root / name).write_text(content, encoding="utf-8")

    monkeypatch.setattr(sidecar_main, "default_curriculum_dir", default_curriculum)
    monkeypatch.setattr(sidecar_main, "overlay_root", tmp_path / "sessions" / "_chapter_overlays")

    with pytest.raises(ValueError):
        sidecar_main._resolve_curriculum_dir(
            "course_1/..",
            {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
        )


def test_resolve_curriculum_dir_legacy_id_writes_to_legacy_chapters_path(tmp_path, monkeypatch):
    default_curriculum = tmp_path / "default_curriculum"
    (default_curriculum / "_templates").mkdir(parents=True, exist_ok=True)
    (default_curriculum / "_templates" / "dynamic_report_template.md").write_text("template", encoding="utf-8")

    bundle_root = tmp_path / "bundle"
    chapter_dir = bundle_root / "legacy_chapter_1"
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [
        ("chapter_context.md", "# chapter"),
        ("task_list.md", "- task"),
        ("task_completion_principles.md", "- principle"),
        ("interaction_protocol.md", "- protocol"),
        ("socratic_vs_direct.md", "- mode"),
    ]:
        (chapter_dir / name).write_text(content, encoding="utf-8")

    monkeypatch.setattr(sidecar_main, "default_curriculum_dir", default_curriculum)
    monkeypatch.setattr(sidecar_main, "overlay_root", tmp_path / "sessions" / "_chapter_overlays")

    resolved = sidecar_main._resolve_curriculum_dir(
        "legacy_chapter_1",
        {"bundle_paths": {"chapter_bundle_path": str(bundle_root)}},
    )
    legacy_target = resolved / "chapters" / "legacy_chapter_1"
    wrong_target = resolved / "courses" / "legacy_course" / "chapters" / "legacy_chapter_1"

    assert legacy_target.exists()
    assert (legacy_target / "chapter_context.md").exists()
    assert not wrong_target.exists()


def test_build_chapter_bundle_script_generates_valid_tar_and_manifest(tmp_path):
    chapter_dir = (
        tmp_path
        / "noise"
        / "courses"
        / "prefix_course"
        / "chapters"
        / "prefix_ch"
        / "curriculum"
        / "courses"
        / "course_1"
        / "chapters"
        / "ch_01"
    )
    chapter_dir.mkdir(parents=True, exist_ok=True)
    for name, content in [
        ("chapter_context.md", "# Intro"),
        ("task_list.md", "- t1"),
        ("task_completion_principles.md", "- p1"),
        ("interaction_protocol.md", "- ip"),
    ]:
        (chapter_dir / name).write_text(content, encoding="utf-8")
    (chapter_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (chapter_dir / "scripts" / "starter_code.py").write_text("print('starter')", encoding="utf-8")
    (chapter_dir / "datasets").mkdir(parents=True, exist_ok=True)
    (chapter_dir / "datasets" / "sample.csv").write_text("a\n1\n", encoding="utf-8")

    output_dir = tmp_path / "dist"
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_chapter_bundle.py"
    completed = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--chapter-dir",
            str(chapter_dir),
            "--output",
            str(output_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "bundle:" in completed.stdout

    bundle_path = output_dir / "chapter_bundle.tar.gz"
    assert bundle_path.exists()

    extract_dir = tmp_path / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    manifest_path = extract_dir / "bundle.manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format_version"] == "bundle-v2"
    assert manifest["bundle_type"] == "chapter"
    assert manifest["chapter"]["course_id"] == "course_1"
    assert manifest["chapter"]["chapter_code"] == "ch_01"
    assert manifest["chapter"]["has_scripts"] is True
    assert manifest["chapter"]["has_datasets"] is True

    file_paths = {entry["path"] for entry in manifest["files"]}
    assert "prompts/chapter_context.md" in file_paths
    assert "scripts/starter_code.py" in file_paths
    assert "datasets/sample.csv" in file_paths
    assert (extract_dir / "prompts" / "chapter_context.md").exists()


def test_restore_content_chapter_files_script_syncs_demo_and_fallback(tmp_path):
    content_curriculum = tmp_path / "content" / "curriculum"
    companion_dir = tmp_path / "content" / "agents" / "companion"
    demo_curriculum = tmp_path / "demo" / "curriculum"

    ch1 = content_curriculum / "courses" / "course_a" / "chapters" / "ch_1"
    ch2 = content_curriculum / "courses" / "course_a" / "chapters" / "ch_2"
    ch1.mkdir(parents=True, exist_ok=True)
    ch2.mkdir(parents=True, exist_ok=True)
    companion_dir.mkdir(parents=True, exist_ok=True)

    demo_ch1 = demo_curriculum / "courses" / "course_a" / "chapters" / "ch_1"
    demo_ch1.mkdir(parents=True, exist_ok=True)
    (demo_ch1 / "interaction_protocol.md").write_text("demo-interaction", encoding="utf-8")
    (demo_ch1 / "consultation_guide.md").write_text("demo-guide", encoding="utf-8")

    (companion_dir / "interaction_protocol.md").write_text("fallback-interaction", encoding="utf-8")
    (companion_dir / "socratic_vs_direct.md").write_text("fallback-socratic", encoding="utf-8")

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "restore_content_chapter_files.py"
    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--content-curriculum",
            str(content_curriculum),
            "--demo-curriculum",
            str(demo_curriculum),
            "--companion-prompts-dir",
            str(companion_dir),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert (ch1 / "interaction_protocol.md").read_text(encoding="utf-8") == "demo-interaction"
    assert (ch1 / "socratic_vs_direct.md").read_text(encoding="utf-8") == "fallback-socratic"
    assert (ch1 / "consultation_guide.md").read_text(encoding="utf-8") == "demo-guide"

    assert (ch2 / "interaction_protocol.md").read_text(encoding="utf-8") == "fallback-interaction"
    assert (ch2 / "socratic_vs_direct.md").read_text(encoding="utf-8") == "fallback-socratic"
