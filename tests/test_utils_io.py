import hashlib
from io import BytesIO

from mri_recon.utils.io import download_file_with_sha256, download_google_drive_file_with_sha256


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


def test_download_google_drive_file_with_sha256_confirms_large_download(tmp_path, monkeypatch):
    warning_html = b"""<!DOCTYPE html><html><body><form id="download-form" action="https://drive.usercontent.google.com/download" method="get"><input type="hidden" name="id" value="file-123"><input type="hidden" name="export" value="download"><input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="uuid-456"></form><title>Google Drive - Virus scan warning</title></body></html>"""
    payload = b"oasis-checkpoint"
    expected_sha256 = hashlib.sha256(payload).hexdigest()
    destination = tmp_path / "oasis.ckpt"
    requested_urls = []

    def fake_urlopen(url, timeout=30):
        requested_urls.append(url)
        if "confirm=t" in url:
            return FakeResponse(payload)
        return FakeResponse(warning_html)

    monkeypatch.setattr("mri_recon.utils.io.urlopen", fake_urlopen)

    download_google_drive_file_with_sha256(
        "file-123",
        destination,
        expected_sha256,
        label="OASIS checkpoint",
        report_interval_mb=1,
    )

    assert destination.read_bytes() == payload
    assert any("confirm=t" in url and "uuid=uuid-456" in url for url in requested_urls)
