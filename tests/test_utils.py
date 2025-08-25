import os
import tarfile
from pathlib import Path

from smt.utils import create_tar_file


def test_create_tar_skips_venv(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    # regular file
    (src / "file.txt").write_text("data")
    # nested directory
    (src / "sub").mkdir()
    (src / "sub" / "subfile.txt").write_text("subdata")
    # .venv directories
    venv_root = src / ".venv"
    venv_root.mkdir()
    (venv_root / "ignored.txt").write_text("ignore")
    nested_venv = src / "sub" / ".venv" / "inner"
    nested_venv.mkdir(parents=True)
    (nested_venv / "ignored.txt").write_text("inner")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path))

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    assert "file.txt" in names
    assert os.path.join("sub", "subfile.txt") in names
    # ensure .venv contents are not included
    assert all(".venv" not in n.split(os.sep) for n in names)


def test_create_tar_with_exclude_patterns_files(tmp_path: Path):
    """ファイルパターンの除外テスト"""
    src = tmp_path / "src"
    src.mkdir()

    # 含まれるべきファイル
    (src / "main.py").write_text("print('hello')")
    (src / "config.json").write_text('{"key": "value"}')

    # 除外されるべきファイル
    (src / "app.log").write_text("log content")
    (src / "temp.tmp").write_text("temporary")
    (src / "compiled.pyc").write_text("bytecode")

    tar_path = tmp_path / "out.tar.gz"
    exclude_patterns = ["*.log", "*.tmp", "*.pyc"]
    create_tar_file(str(src), str(tar_path), exclude_patterns)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 含まれるファイル
    assert "main.py" in names
    assert "config.json" in names

    # 除外されるファイル
    assert "app.log" not in names
    assert "temp.tmp" not in names
    assert "compiled.pyc" not in names


def test_create_tar_with_exclude_patterns_directories(tmp_path: Path):
    """ディレクトリパターンの除外テスト"""
    src = tmp_path / "src"
    src.mkdir()

    # 含まれるべきディレクトリとファイル
    (src / "source").mkdir()
    (src / "source" / "main.py").write_text("print('hello')")

    # 除外されるべきディレクトリとファイル
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "main.cpython-311.pyc").write_text("bytecode")

    (src / ".git").mkdir()
    (src / ".git" / "config").write_text("git config")

    (src / "node_modules").mkdir()
    (src / "node_modules" / "package.json").write_text('{"name": "test"}')

    tar_path = tmp_path / "out.tar.gz"
    exclude_patterns = ["__pycache__", ".git", "node_modules"]
    create_tar_file(str(src), str(tar_path), exclude_patterns)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 含まれるファイル
    assert os.path.join("source", "main.py") in names

    # 除外されるディレクトリとファイル
    assert not any("__pycache__" in name for name in names)
    assert not any(".git" in name for name in names)
    assert not any("node_modules" in name for name in names)


def test_create_tar_with_exclude_patterns_nested_paths(tmp_path: Path):
    """ネストしたパスパターンの除外テスト"""
    src = tmp_path / "src"
    src.mkdir()

    # 含まれるべきファイル
    (src / "tests").mkdir()
    (src / "tests" / "test_main.py").write_text("def test_main(): pass")
    (src / "tests" / "unit").mkdir()
    (src / "tests" / "unit" / "test_utils.py").write_text("def test_utils(): pass")

    # 除外されるべきファイル
    (src / "tests" / "fixtures").mkdir()
    (src / "tests" / "fixtures" / "data.json").write_text('{"test": "data"}')

    (src / "docs" / "build").mkdir(parents=True)
    (src / "docs" / "build" / "index.html").write_text("<html></html>")
    (src / "docs" / "source").mkdir(parents=True)
    (src / "docs" / "source" / "index.rst").write_text("Documentation")

    tar_path = tmp_path / "out.tar.gz"
    exclude_patterns = ["tests/fixtures", "docs/build"]
    create_tar_file(str(src), str(tar_path), exclude_patterns)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 含まれるファイル
    assert os.path.join("tests", "test_main.py") in names
    assert os.path.join("tests", "unit", "test_utils.py") in names
    assert os.path.join("docs", "source", "index.rst") in names

    # 除外されるファイル
    assert not any("tests/fixtures" in name for name in names)
    assert not any("docs/build" in name for name in names)


def test_create_tar_with_mixed_exclude_patterns(tmp_path: Path):
    """ファイルとディレクトリの混在した除外パターンテスト"""
    src = tmp_path / "src"
    src.mkdir()

    # 通常のプロジェクト構造を作成
    (src / "src").mkdir()
    (src / "src" / "main.py").write_text("print('hello')")
    (src / "src" / "utils.py").write_text("def helper(): pass")

    (src / "tests").mkdir()
    (src / "tests" / "test_main.py").write_text("def test(): pass")

    # 除外対象
    (src / "__pycache__").mkdir()
    (src / "__pycache__" / "main.cpython-311.pyc").write_text("bytecode")

    (src / ".pytest_cache").mkdir()
    (src / ".pytest_cache" / "cache.json").write_text("{}")

    (src / "debug.log").write_text("debug info")
    (src / "error.log").write_text("error info")
    (src / "temp.tmp").write_text("temporary")

    (src / "coverage.xml").write_text("<coverage></coverage>")

    tar_path = tmp_path / "out.tar.gz"
    exclude_patterns = ["__pycache__", ".pytest_cache", "*.log", "*.tmp", "coverage.*"]
    create_tar_file(str(src), str(tar_path), exclude_patterns)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 含まれるファイル
    assert os.path.join("src", "main.py") in names
    assert os.path.join("src", "utils.py") in names
    assert os.path.join("tests", "test_main.py") in names

    # 除外されるファイル・ディレクトリ
    assert not any("__pycache__" in name for name in names)
    assert not any(".pytest_cache" in name for name in names)
    assert "debug.log" not in names
    assert "error.log" not in names
    assert "temp.tmp" not in names
    assert "coverage.xml" not in names


def test_create_tar_exclude_patterns_none_defaults_to_empty_list(tmp_path: Path):
    """exclude_patternsがNoneの場合のデフォルト動作テスト"""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("print('hello')")
    (src / "test.log").write_text("log content")  # パターンなしなので含まれる

    # .venvは常に除外される
    venv = src / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("lib content")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path), exclude_patterns=None)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 通常のファイルは含まれる
    assert "main.py" in names
    assert "test.log" in names  # 除外パターンがないので含まれる

    # .venvは常に除外
    assert not any(".venv" in name for name in names)


def test_create_tar_exclude_patterns_empty_list(tmp_path: Path):
    """exclude_patternsが空リストの場合のテスト"""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("print('hello')")
    (src / "test.log").write_text("log content")

    # .venvは常に除外される
    venv = src / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("lib content")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path), exclude_patterns=[])

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 通常のファイルは含まれる
    assert "main.py" in names
    assert "test.log" in names

    # .venvは常に除外
    assert not any(".venv" in name for name in names)
