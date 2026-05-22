# ArtifactLab

[![CI](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/MatthiasLen/mri_recon/actions/workflows/ci.yml)

ArtifactLab is a small Python package for studying how k-space distortions affect MRI reconstruction. It provides distortion operators, reference reconstruction methods, and example inference workflows for fastMRI single-coil knee data and OASIS brain MRI.

## Installation

The project is managed with `uv`.

```bash
uv sync
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

## Implemented Reconstruction Algorithms

| Class | Selector(s) | Family | Summary |
| --- | --- | --- | --- |
| `ZeroFilledReconstructor` | `zero-filled` | Direct baseline | Returns the adjoint reconstruction, i.e. the standard zero-filled inverse FFT baseline. |
| `ConjugateGradientReconstructor` | `conjugate-gradient` | Classical iterative | Uses `physics.A_dagger(...)` to solve the inverse problem with conjugate-gradient style least-squares reconstruction. |
| `TVPGDReconstructor` | `tv-pgd` | Variational iterative | Proximal gradient descent with an L2 data term and total-variation prior. |
| `WaveletFISTAReconstructor` | `wavelet-fista` | Variational iterative | FISTA with an L1 wavelet prior for sparse regularization in a wavelet basis. |
| `TVFISTAReconstructor` | `tv-fista` | Variational iterative | FISTA with total-variation regularization. |
| `TVPDHGReconstructor` | `tv-pdhg` | Variational iterative | Primal-dual hybrid gradient / Chambolle-Pock optimization with total-variation regularization. |
| `RAMReconstructor` | `ram` | Deep learning | Wrapper around the DeepInverse RAM model, with input normalization based on the adjoint reconstruction. |
| `DeepImagePriorReconstructor` | `dip` | Deep learning | Deep Image Prior reconstruction using an untrained convolutional decoder optimized at inference time. |
| `FastMRISinglecoilUnetReconstructor` | `unet-fastmri` | Deep learning | Pretrained fastMRI single-coil knee U-Net. The wrapper returns a magnitude image with a zero imaginary channel. |
| `OASISSinglecoilUnetReconstructor` | `unet-oasis-acceleration4`, `unet-oasis-acceleration8`, `unet-oasis-acceleration10` | Deep learning | Trained OASIS single-coil U-Net checkpoints for the listed training accelerations, reusing the shared fastMRI-derived U-Net module. |

## Implemented Distortions

| Class | Selector name | Family | Targeted Image Property | Summary |
| --- | --- | --- | --- | --- |
| `BaseDistortion` | `None` | Identity | `n/a` | Leaves the k-space unchanged and serves as the no-distortion baseline. |
| `SelfAdjointMultiplicativeMaskDistortion` | `n/a` | Abstract base | `n/a` | Base class for self-adjoint distortions that apply a real-valued elementwise multiplicative mask; subclasses implement `_mask`. |
| `IsotropicResolutionReduction` | `Isotropic LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning) | Applies a circular low-pass mask in k-space to remove high frequencies isotropically. |
| `AnisotropicResolutionReduction` | `Anisotropic LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning) | Applies an axis-aligned rectangular low-pass mask with separate cutoffs along `kx` and `ky`. |
| `HannTaperResolutionReduction` | `Hann taper LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning), RoI Homogeneity (Glancing) | Applies a circular low-pass mask with a raised-cosine transition band to soften the cutoff. |
| `KaiserTaperResolutionReduction` | `Kaiser taper LP` | Resolution loss | Sharpness (Glancing), Edges (Scanning), RoI Homogeneity (Glancing) | Applies a circular low-pass mask with a Kaiser transition band for adjustable cutoff smoothness. |
| `CartesianUndersampling` | `Cartesian undersampling` | Acquisition undersampling | Local Signal Preservation (Scanning) | Simulates Cartesian acquisition undersampling with optional contiguous ACS center retention plus uniform-random, variable-density-random, or equispaced peripheral sampling. |
| `PartialFourierDistortion` | `Partial Fourier` | Acquisition undersampling | Local Signal Preservation (Scanning) | Simulates partial Fourier acquisition by retaining a contiguous asymmetric region of k-space along one encoding axis while preserving a centered low-frequency block. |
| `RadialHighPassEmphasisDistortion` | `Radial high-pass emphasis` | Sharpening | Sharpness (Glancing), Edges (Scanning), Noise Level (Glancing) | Applies a radial gain mask that increasingly boosts high-frequency k-space content toward the sampled edge. |
| `GaussianKspaceBiasField` | `Gaussian bias field` | Intensity non-uniformity | Intensity Uniformity (Glancing/Full Image) | Applies a centered smooth multiplicative Gaussian gain field in k-space. |
| `OffCenterAnisotropicGaussianKspaceBiasField` | `Off-center anisotropic Gaussian bias field` | Intensity non-uniformity | Intensity Uniformity (Glancing/Full Image) | Applies an off-center anisotropic Gaussian gain field in k-space with separate widths along `kx` and `ky`. |
| `GaussianNoiseDistortion` | `Gaussian noise` | Noise | Noise Level (Glancing) | Adds independent zero-mean Gaussian noise to the stored real and imaginary k-space channels. |
| `TranslationMotionDistortion` | `Translation motion` | Motion | Local Signal Preservation (Scanning) | Applies a rigid in-plane translation as a unit-modulus phase ramp in k-space. |
| `RotationalMotionDistortion` | `Rotational motion` | Motion | Local Signal Preservation (Scanning) | Applies a rigid in-plane rotation about the image center by resampling centered Cartesian k-space. |
| `SegmentedRotationalMotionDistortion` | `Segmented rotational motion` | Motion | Local Signal Preservation (Scanning) | Splits Cartesian k-space into acquisition segments and stitches segment-specific centered k-space rotations into one inconsistent scan. |
| `SegmentedTranslationMotionDistortion` | `Segmented translation motion` | Motion | Local Signal Preservation (Scanning) | Splits Cartesian k-space into acquisition segments and applies a different translation phase ramp to each segment. |
| `PhaseEncodeGhostingDistortion` | `Phase-encode ghosting` | Ghosting | Local Signal Preservation (Scanning) | Applies periodic line-wise phase and magnitude inconsistency to create phase-encode ghost replicas. |

The example script also exposes several named `CartesianUndersampling` variants, including variable-density random, uniform random, and equispaced masks.

## Inference Examples

The script [examples/fastmri_inference_plot.py](examples/fastmri_inference_plot.py) writes comparison figures to `reports/fastmri_inference_plot` or `reports/oasis_inference_plot`.

Run FastMRI inference with the packaged fastMRI U-Net:

```bash
uv run python examples/fastmri_inference_plot.py --source /path/to/fastmri/singlecoil_val --dataset fastmri --algorithm unet-fastmri
```

Run the same FastMRI measurement through an OASIS-trained U-Net:

```bash
uv run python examples/fastmri_inference_plot.py --source /path/to/fastmri/singlecoil_val --dataset fastmri --algorithm unet-oasis-acceleration8
```

Run OASIS inference. The packaged OASIS split CSV and checkpoint are downloaded when needed:

```bash
uv run python examples/fastmri_inference_plot.py --source /path/to/oasis_cross_sectional_data --dataset oasis --algorithm unet-oasis-acceleration4
```

Supported packaged U-Net selectors are `unet-fastmri`, `unet-oasis-acceleration4`, `unet-oasis-acceleration8`, and `unet-oasis-acceleration10`. `unet-fastmri` is intentionally rejected on the OASIS dataset.

## Pre-commit

Install the hook and run the repository checks:

```bash
uv sync
uv run pre-commit install
uv run pre-commit run --all-files
```

CI runs the same `pre-commit` checks and the test suite with `uv run pytest`.

## Contributing

1. Install and run `pre-commit` before opening a pull request.
2. Add NumPy-style docstrings to public functions, methods, and classes.
3. If you add a reconstructor or distortion, update the corresponding README table.
4. Add or update tests in `tests/` for new behaviour.
