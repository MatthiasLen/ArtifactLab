import hashlib
from io import BytesIO

from mri_recon.utils.io import download_file_with_sha256


class FakeResponse:
    def __init__(self, payload: bytes):
        self._buffer = BytesIO(payload)
        self.headers = {"Content-Length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        return self._buffer.read(size)


def test_download_file_with_sha256_moves_temp_file_after_close(tmp_path, monkeypatch):
    payload = b"test-payload-for-fastmri-download"
    destination = tmp_path / "checkpoint.pt"
    expected_sha256 = hashlib.sha256(payload).hexdigest()

    monkeypatch.setattr(
        "mri_recon.utils.io.urlopen",
        lambda url, timeout=30: FakeResponse(payload),
    )

    download_file_with_sha256(
        "https://example.com/checkpoint.pt",
        destination,
        expected_sha256,
        label="checkpoint",
        report_interval_mb=1,
    )

    assert destination.read_bytes() == payload
    assert list(tmp_path.glob("*.tmp")) == []
