import hashlib
import sys
from pathlib import Path

import pytest
from scripts import download_model


@pytest.mark.parametrize("failure", ["network", "checksum"])
def test_download_failure_preserves_model_and_cleans_partial(tmp_path, monkeypatch, failure):
    output = tmp_path / "model.ckpt"
    output.write_bytes(b"existing model")
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--output", str(output)])
    def download(url, destination):
        Path(destination).write_bytes(b"incomplete download")
        if failure == "network":
            raise OSError("interrupted connection")
    monkeypatch.setattr(download_model.urllib.request, "urlretrieve", download)
    with pytest.raises((OSError, RuntimeError)):
        download_model.main()
    assert output.read_bytes() == b"existing model"
    assert list(tmp_path.iterdir()) == [output]


def test_verified_download_replaces_result(tmp_path, monkeypatch):
    output = tmp_path / "model.ckpt"
    output.write_bytes(b"previous model")
    payload = b"verified model fixture"
    monkeypatch.setattr(download_model, "MODEL_SHA256", hashlib.sha256(payload).hexdigest())
    monkeypatch.setattr(download_model.urllib.request, "urlretrieve",
                        lambda url, destination: Path(destination).write_bytes(payload))
    monkeypatch.setattr(sys, "argv", ["download_model.py", "--output", str(output)])
    download_model.main()
    assert output.read_bytes() == payload
    assert list(tmp_path.iterdir()) == [output]
