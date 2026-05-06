# mri_recon

[![CI](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml)

MRI reconstruction playground for the MRI Metrics project.

## FastMRI UNet Reconstructor

The package includes `FastMRISinglecoilUnetReconstructor` in `mri_recon.reconstruction`.
It wraps the pretrained fastMRI single-coil knee U-Net and applies the same per-slice
magnitude normalization used during fastMRI training before rescaling the predicted
image back to the original adjoint-image intensity range.

By default the checkpoint is stored at the repository root as `knee_sc_leaderboard_state_dict.pt`.
If the file is missing, the reconstructor downloads it from the official fastMRI URL,
verifies its SHA256 checksum, and then loads the weights with `torch.load(..., weights_only=True)`.
You can also pass a local checkpoint path explicitly through the `state_dict_file` argument.

The model predicts a magnitude image, so the wrapper returns that output in the real channel
and fills the imaginary channel with zeros.

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

GitHub Actions runs the same `pre-commit` command in CI and also runs the test suite with `uv run pytest`.
