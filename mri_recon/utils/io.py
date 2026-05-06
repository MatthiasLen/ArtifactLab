import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen


def matches_sha256(path: Path, expected_sha256: str) -> bool:
    if not path.exists():
        return False

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected_sha256


def format_megabytes(num_bytes: int) -> str:
    return f"{num_bytes / (1024 * 1024):.1f} MB"


def download_file_with_sha256(
    url: str,
    destination: Path,
    expected_sha256: str,
    *,
    label: str = "file",
    report_interval_mb: int = 25,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {label} from {url} to {destination}. This may take a moment.")

    report_interval = report_interval_mb * 1024 * 1024
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", delete=False, dir=destination.parent, suffix=".tmp"
        ) as handle:
            tmp_path = Path(handle.name)

            with urlopen(url, timeout=30) as response:
                total_size = response.headers.get("Content-Length")
                total_size = int(total_size) if total_size is not None else None
                downloaded = 0
                next_report = report_interval

                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    handle.write(chunk)
                    downloaded += len(chunk)

                    should_report = downloaded >= next_report
                    if total_size is not None and downloaded == total_size:
                        should_report = True

                    if should_report:
                        if total_size is None:
                            print(f"Downloaded {format_megabytes(downloaded)} of {label}...")
                        else:
                            print(
                                f"Downloaded {format_megabytes(downloaded)} / "
                                f"{format_megabytes(total_size)} "
                                f"({100 * downloaded / total_size:.1f}%)"
                            )
                        next_report += report_interval

        if not matches_sha256(tmp_path, expected_sha256):
            raise ValueError(f"Downloaded file failed SHA256 verification: {destination}")

        tmp_path.replace(destination)
        print(f"{label.capitalize()} saved to {destination}.")
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
