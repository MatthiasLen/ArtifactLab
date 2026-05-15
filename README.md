# ArtifactLab

[![CI](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml)

Reconstruction playground for the MRI Recon Metrics Reloaded workgroup.

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
| `FastMRISinglecoilUnetReconstructor` | `unet` | Deep learning | Wrapper around the pretrained fastMRI single-coil U-Net, returning a magnitude-based reconstruction with a zero imaginary channel. |
| `OASISSinglecoilUnetReconstructor` | `oasis-unet` | Deep learning | Wrapper around a trained OASIS single-coil U-Net checkpoint, reusing the shared fastMRI-derived U-Net module. |

## Implemented Distortions

| Class | Selector name | Family | Targeted Image Property | Summary |
| --- | --- | --- | --- | --- |
| `BaseDistortion` | `None` | Identity | | Leaves the k-space unchanged and serves as the no-distortion baseline. |
| `SelfAdjointMultiplicativeMaskDistortion` | `None` | Abstract base | | Super class for self-adjoint distortions that apply a real-valued elementwise multiplicative mask; subclasses implement `_mask`. |
| `IsotropicResolutionReduction` | `Isotropic LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning) | Applies a circular low-pass mask in k-space to remove high frequencies isotropically. |
| `AnisotropicResolutionReduction` | `Anisotropic LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning) | Applies an axis-aligned rectangular low-pass mask with separate cutoffs along `kx` and `ky`. |
| `HannTaperResolutionReduction` | `Hann taper LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning), RoI Homogeneity (Glancing) | Applies a circular low-pass mask with a raised-cosine transition band to soften the cutoff. |
| `KaiserTaperResolutionReduction` | `Kaiser taper LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning), RoI Homogeneity (Glancing) | Applies a circular low-pass mask with a Kaiser transition band for adjustable cutoff smoothness. |
| `CartesianUndersampling` | `Cartesian undersampling` | Acquisition undersampling | Local Signal Preservation (Scanning) | Simulates Cartesian acquisition undersampling with optional contiguous ACS center retention plus uniform-random, variable-density-random, or equispaced peripheral sampling. |
| `PartialFourierDistortion` | `Partial Fourier` | Acquisition undersampling | Local Signal Preservation (Scanning) | Simulates partial Fourier acquisition by retaining a contiguous asymmetric region of k-space along one encoding axis while preserving a centered low-frequency block. |
| `RadialHighPassEmphasisDistortion` | `Radial high-pass emphasis` | Sharpening |  Sharpness (Glancing), Edges (Scanning), Noise Level (Glancing) | Applies a radial gain mask that increasingly boosts high-frequency k-space content toward the sampled edge. |
| `GaussianKspaceBiasField` | `Gaussian bias field` | Intensity non-uniformity | Intensity Uniformity (Glancing/Full Image) | Applies a centered smooth multiplicative Gaussian gain field in k-space. |
| `OffCenterAnisotropicGaussianKspaceBiasField` | `Off-center anisotropic Gaussian bias field` | Intensity non-uniformity | Intensity Uniformity (Glancing/Full Image) | Applies an off-center anisotropic Gaussian gain field in k-space with separate widths along `kx` and `ky`. |
| `GaussianNoiseDistortion` | `Gaussian noise` | Noise | Noise Level (Glancing) | Adds independent zero-mean Gaussian noise to the stored real and imaginary k-space channels. |
| `TranslationMotionDistortion` | `Translation motion` | Motion | Local Signal Preservation (Scanning) |Applies a rigid in-plane translation as a unit-modulus phase ramp in k-space. |
| `RotationalMotionDistortion` | `Rotational motion` | Motion | Local Signal Preservation (Scanning) | Applies a rigid in-plane rotation about the image center by resampling centered Cartesian k-space. |
| `SegmentedRotationalMotionDistortion` | `Segmented rotational motion` | Motion | Local Signal Preservation (Scanning) | Splits Cartesian k-space into acquisition segments and stitches segment-specific centered k-space rotations into one inconsistent scan. |
| `SegmentedTranslationMotionDistortion` | `Segmented translation motion` | Motion | Local Signal Preservation (Scanning) | Splits Cartesian k-space into acquisition segments and applies a different translation phase ramp to each segment. |
| `PhaseEncodeGhostingDistortion` | `Phase-encode ghosting` | Ghosting | Local Signal Preservation (Scanning) | Applies periodic line-wise phase and magnitude inconsistency to create phase-encode ghost replicas. |


## Distortions Possibly Planned
- Rician Noise (Noise Level (Glancing))
- Modification of sensitivity map (RoI Homogeneity (Glancing))


## uv Environment Notes

This project uses `uv.lock` and pins PyTorch through `uv` package indexes in `pyproject.toml`.

On Windows and Linux, `uv sync` installs the CUDA 12.8 PyTorch wheels. On macOS, it falls back to CPU wheels.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Inference Examples

Run the FastMRI plotting example with local FastMRI k-space files:

```bash
python examples/fastmri_inference_plot.py --source /path/to/fastmri/singlecoil_val --dataset fastmri --algorithm unet-fastmri
```

Run the same FastMRI data through the packaged OASIS U-Net by choosing an explicit OASIS checkpoint variant. The example adapts the FastMRI measurements to the centered OASIS FFT convention automatically:

```bash
python examples/fastmri_inference_plot.py --source /path/to/fastmri/singlecoil_val --dataset fastmri --algorithm unet-oasis-acceleration8
```

Run the same lightweight example on OASIS data. The packaged OASIS split CSV and U-Net checkpoint are downloaded automatically when missing:

```bash
python examples/fastmri_inference_plot.py --source /path/to/oasis_cross_sectional_data --dataset oasis --algorithm unet-oasis-acceleration4
```

Supported explicit U-Net algorithms are `unet-fastmri`, `unet-oasis-acceleration4`, `unet-oasis-acceleration8`, and `unet-oasis-acceleration10`. `unet-fastmri` on the OASIS dataset is intentionally rejected.

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
