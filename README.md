# mri_recon

MRI reconstruction playground for the MRI Metrics project.

## uv Environment Notes

This project uses `uv.lock` and pins PyTorch through `uv` package indexes in `pyproject.toml`.

On Windows and Linux, `uv sync` installs the CUDA 12.8 PyTorch wheels. On macOS, it falls back to CPU wheels.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Pre-commit

Install the local tooling and register the git hook:

```bash
uv sync
uv run pre-commit install
```

Run the hook suite manually across the repository:

```bash
uv run pre-commit run --all-files
```

GitHub Actions runs the same command in CI, so the local hook and CI stay aligned.
