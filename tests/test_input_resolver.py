import pytest
import os
import tempfile
from io_flow.input_resolver import resolve_image_paths

def test_resolve_image_paths_single():
    cfg = {"image_path": "/fake/path/image1.jpg"}
    with pytest.raises(FileNotFoundError):
        # Since the file doesn't exist, it should raise
        resolve_image_paths(cfg)
        
def test_resolve_image_paths_existing():
    with tempfile.NamedTemporaryFile(suffix=".jpg") as tmp:
        cfg = {"image_path": tmp.name}
        paths = resolve_image_paths(cfg)
        assert len(paths) == 1
        assert paths[0] == os.path.abspath(tmp.name)

def test_resolve_image_paths_glob():
    with tempfile.TemporaryDirectory() as tmpdir:
        f1 = os.path.join(tmpdir, "img1.JPG")
        f2 = os.path.join(tmpdir, "img2.JPG")
        open(f1, 'a').close()
        open(f2, 'a').close()
        
        cfg = {"image_glob": os.path.join(tmpdir, "*.JPG")}
        paths = resolve_image_paths(cfg)
        
        assert len(paths) == 2
        assert set(paths) == {os.path.abspath(f1), os.path.abspath(f2)}
