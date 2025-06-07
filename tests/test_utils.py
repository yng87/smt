import os
import sys
import types
from pathlib import Path
import tarfile

# Provide dummy sagemaker module to satisfy import in smt.utils
dummy_sm = types.ModuleType("sagemaker")
dummy_sm.session = types.SimpleNamespace(Session=lambda *_, **__: None)
sys.modules.setdefault("sagemaker", dummy_sm)

# Provide minimal pydantic replacement so importing smt.utils doesn't require
# installing the real dependency
dummy_pydantic = types.ModuleType("pydantic")

class DummyBaseModel:
    pass

def dummy_field(*_args, **_kwargs):
    return None

dummy_pydantic.BaseModel = DummyBaseModel
dummy_pydantic.Field = dummy_field

sys.modules.setdefault("pydantic", dummy_pydantic)

from smt.utils import create_tar_file


def test_create_tar_skips_venv(tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    # regular file
    (src / "file.txt").write_text("data")
    # nested directory
    (src / "sub" ).mkdir()
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

