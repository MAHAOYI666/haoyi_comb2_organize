from __future__ import annotations

from pathlib import Path

import pytest

import comboRunner


def test_resolve_config_prefers_explicit_config(tmp_path: Path) -> None:
    script_dir = tmp_path / "model"
    script_dir.mkdir()
    explicit_config = tmp_path / "explicit.xml"
    explicit_config.write_text("<config />", encoding="utf-8")

    assert comboRunner._resolve_config(script_dir, str(explicit_config), None) == explicit_config.resolve()


def test_resolve_config_defaults_to_script_dir_config(tmp_path: Path) -> None:
    script_dir = tmp_path / "model"
    script_dir.mkdir()
    default_config = script_dir / "config.xml"
    default_config.write_text("<config />", encoding="utf-8")

    assert comboRunner._resolve_config(script_dir, None, None) == default_config.resolve()


def test_find_repo_root_from_parent(tmp_path: Path) -> None:
    repo_root = tmp_path / "comb2_organize"
    model_dir = repo_root / "moe"
    model_dir.mkdir(parents=True)
    (repo_root / "runCombo.py").write_text("", encoding="utf-8")
    (repo_root / "config.py").write_text("", encoding="utf-8")

    assert comboRunner._find_repo_root(model_dir) == repo_root.resolve()


def test_find_repo_root_from_sibling_comb2_organize(tmp_path: Path) -> None:
    repo_root = tmp_path / "comb2_organize"
    model_dir = tmp_path / "0612.moe.test"
    repo_root.mkdir()
    model_dir.mkdir()
    (repo_root / "runCombo.py").write_text("", encoding="utf-8")
    (repo_root / "config.py").write_text("", encoding="utf-8")

    assert comboRunner._find_repo_root(model_dir) == repo_root.resolve()


def test_find_repo_root_reports_missing_root(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="comb2_organize root"):
        comboRunner._find_repo_root(tmp_path)
