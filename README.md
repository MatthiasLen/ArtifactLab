# mri_recon

[![CI](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml)

MRI reconstruction playground for the MRI Metrics project.

## Implemented Reconstruction Algorithms

| Class | Selector name | Family | Summary |
| --- | --- | --- | --- |
| `ZeroFilledReconstructor` | `zero-filled` | Direct baseline | Returns the adjoint reconstruction, i.e. the standard zero-filled inverse FFT baseline. |
| `ConjugateGradientReconstructor` | `conjugate-gradient` | Classical iterative | Uses `physics.A_dagger(...)` to solve the inverse problem with conjugate-gradient style least-squares reconstruction. |
| `TVPGDReconstructor` | `tv-pgd` | Variational iterative | Proximal gradient descent with an L2 data term and total-variation prior. |
| `WaveletFISTAReconstructor` | `wavelet-fista` | Variational iterative | FISTA with an L1 wavelet prior for sparse regularization in a wavelet basis. |
| `TVFISTAReconstructor` | `tv-fista` | Variational iterative | FISTA with total-variation regularization. |
| `TVPDHGReconstructor` | `tv-pdhg` | Variational iterative | Primal-dual hybrid gradient / Chambolle-Pock optimization with total-variation regularization. |
| `RAMReconstructor` | `ram` | Deep learning | Wrapper around the DeepInverse RAM model, with input normalization based on the adjoint reconstruction. |
| `DeepImagePriorReconstructor` | `dip` | Deep learning | Deep Image Prior reconstruction using an untrained convolutional decoder optimized at inference time. |

## Implemented Distortions

| Class | Selector name | Family | Summary |
| --- | --- | --- | --- |
| `BaseDistortion` | `None` | Identity | Leaves the k-space unchanged and serves as the no-distortion baseline. |
| `IsotropicResolutionReduction` | `Isotropic LP` | Resolution loss | Applies a circular low-pass mask in k-space to remove high frequencies isotropically. |
| `AnisotropicResolutionReduction` | `Anisotropic LP` | Resolution loss | Applies an axis-aligned rectangular low-pass mask with separate cutoffs along `kx` and `ky`. |
| `HannTaperResolutionReduction` | `Hann taper LP` | Resolution loss | Applies a circular low-pass mask with a raised-cosine transition band to soften the cutoff. |
| `KaiserTaperResolutionReduction` | `Kaiser taper LP` | Resolution loss | Applies a circular low-pass mask with a Kaiser transition band for adjustable cutoff smoothness. |
| `GaussianKspaceBiasField` | `Gaussian bias field` | Intensity non-uniformity | Applies a centered smooth multiplicative Gaussian gain field in k-space. |
| `OffCenterAnisotropicGaussianKspaceBiasField` | `Off-center anisotropic Gaussian bias field` | Intensity non-uniformity | Applies an off-center anisotropic Gaussian gain field in k-space with separate widths along `kx` and `ky`. |
| `GaussianNoiseDistortion` | `Gaussian noise` | Noise | Adds independent zero-mean Gaussian noise to the stored real and imaginary k-space channels. |
| `TranslationMotionDistortion` | `Translation motion` | Motion | Applies a rigid in-plane translation as a unit-modulus phase ramp in k-space. |
| `SegmentedTranslationMotionDistortion` | `Segmented translation motion` | Motion | Splits Cartesian k-space into acquisition segments and applies a different translation phase ramp to each segment. |
| `PhaseEncodeGhostingDistortion` | `Phase-encode ghosting` | Ghosting | Applies periodic line-wise phase and magnitude inconsistency to create phase-encode ghost replicas. |

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

## Contributing

1. **Pre-commit hooks** – install and run them before pushing (see [Pre-commit](#pre-commit) above). CI enforces the same checks.
2. **Docstrings** – add a NumPy-style docstring to every public function, method, and class. Include a one-line summary, `Parameters`, and `Returns` sections where applicable.
3. **README updates** – if you add a new reconstructor or distortion, append a row to the corresponding table. Keep descriptions concise (one sentence).
4. **Tests** – add or update tests under `tests/` for any new behaviour. Run the full suite with `uv run pytest` before opening a PR.
5. **Branching** – open a feature branch, keep commits focused, and open a pull request against `main`.
