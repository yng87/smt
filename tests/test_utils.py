import os
import tarfile
from pathlib import Path

from smt.utils import create_tar_file, substitute_variables


def test_create_tar_includes_venv_by_default(tmp_path: Path):
    """Note: This test validates that .venv is NOT automatically excluded (changed behavior)"""
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
    (venv_root / "included.txt").write_text("included")
    nested_venv = src / "sub" / ".venv" / "inner"
    nested_venv.mkdir(parents=True)
    (nested_venv / "included.txt").write_text("inner")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path))

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    assert "file.txt" in names
    assert os.path.join("sub", "subfile.txt") in names
    # .venv contents are now INCLUDED (changed behavior)
    assert ".venv/included.txt" in names
    assert os.path.join("sub", ".venv", "inner", "included.txt") in names


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

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path), exclude_patterns=None)

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # すべてのファイルが含まれる
    assert "main.py" in names
    assert "test.log" in names


def test_create_tar_exclude_patterns_empty_list(tmp_path: Path):
    """exclude_patternsが空リストの場合のテスト"""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("print('hello')")
    (src / "test.log").write_text("log content")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path), exclude_patterns=[])

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # すべてのファイルが含まれる
    assert "main.py" in names
    assert "test.log" in names


def test_create_tar_can_exclude_venv_explicitly(tmp_path: Path):
    """明示的に.venvを除外パターンに指定した場合のテスト"""
    src = tmp_path / "src"
    src.mkdir()

    (src / "main.py").write_text("print('hello')")

    # .venvディレクトリを作成
    venv = src / ".venv"
    venv.mkdir()
    (venv / "lib.py").write_text("lib content")

    tar_path = tmp_path / "out.tar.gz"
    create_tar_file(str(src), str(tar_path), exclude_patterns=[".venv"])

    with tarfile.open(tar_path) as tar:
        names = tar.getnames()

    # 通常のファイルは含まれる
    assert "main.py" in names

    # 明示的に除外した.venvは含まれない
    assert not any(".venv" in name for name in names)


def test_substitute_variables_string():
    """文字列での変数展開テスト"""
    variables = {"run_id": "20231225T120000123456"}

    # 単純な変数展開
    result = substitute_variables("s3://bucket/checkpoints/${run_id}", variables)
    assert result == "s3://bucket/checkpoints/20231225T120000123456"

    # 複数の変数
    variables["bucket"] = "my-bucket"
    result = substitute_variables("s3://${bucket}/checkpoints/${run_id}", variables)
    assert result == "s3://my-bucket/checkpoints/20231225T120000123456"

    # 変数がない場合
    result = substitute_variables("s3://bucket/path", variables)
    assert result == "s3://bucket/path"


def test_substitute_variables_dict():
    """辞書での変数展開テスト"""
    variables = {"run_id": "20231225T120000123456"}

    config = {
        "checkpoint_s3_uri": "s3://bucket/checkpoints/${run_id}",
        "output_path": "s3://bucket/outputs/${run_id}",
        "instance_type": "ml.m5.large",
    }

    result = substitute_variables(config, variables)

    assert (
        result["checkpoint_s3_uri"] == "s3://bucket/checkpoints/20231225T120000123456"
    )
    assert result["output_path"] == "s3://bucket/outputs/20231225T120000123456"
    assert result["instance_type"] == "ml.m5.large"  # 変数がない値はそのまま


def test_substitute_variables_nested():
    """ネストした構造での変数展開テスト"""
    variables = {"run_id": "20231225T120000123456"}

    config = {
        "estimator_config": {
            "checkpoint_s3_uri": "s3://bucket/checkpoints/${run_id}",
            "hyperparameters": {"model_dir": "/opt/ml/model/${run_id}", "epochs": 10},
        },
        "paths": ["s3://bucket/data/${run_id}", "s3://bucket/logs/${run_id}"],
    }

    result = substitute_variables(config, variables)

    assert (
        result["estimator_config"]["checkpoint_s3_uri"]
        == "s3://bucket/checkpoints/20231225T120000123456"
    )
    assert (
        result["estimator_config"]["hyperparameters"]["model_dir"]
        == "/opt/ml/model/20231225T120000123456"
    )
    assert result["estimator_config"]["hyperparameters"]["epochs"] == 10
    assert result["paths"][0] == "s3://bucket/data/20231225T120000123456"
    assert result["paths"][1] == "s3://bucket/logs/20231225T120000123456"


def test_substitute_variables_non_string():
    """文字列以外の値での変数展開テスト"""
    variables = {"run_id": "20231225T120000123456"}

    # 数値、ブール値、Noneは変更されない
    assert substitute_variables(42, variables) == 42
    assert substitute_variables(True, variables) is True
    assert substitute_variables(None, variables) is None
