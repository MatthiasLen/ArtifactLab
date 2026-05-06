import hashlib
import tempfile
from pathlib import Path
from urllib.request import urlopen

from tqdm.auto import tqdm


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

    chunk_size = 1024 * 1024
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
                with tqdm(
                    total=total_size,
                    desc=f"Downloading {label}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    miniters=max(1, report_interval // chunk_size),
                    disable=False,
                ) as progress:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        handle.write(chunk)
                        progress.update(len(chunk))

        if not matches_sha256(tmp_path, expected_sha256):
            raise ValueError(f"Downloaded file failed SHA256 verification: {destination}")

        tmp_path.replace(destination)
        print(f"{label.capitalize()} saved to {destination}.")
    except Exception:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
        raise
