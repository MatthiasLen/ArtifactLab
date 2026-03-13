# mri_recon

MRI reconstruction playground for the MRI Metrics project with:

- Dataset wrapper for FastMRI slices (`FastMRIDataset`)
- Reconstruction algorithms (classic + undersampled + DeepInverse wrapper)
- Image/k-space distortion operators (resolution, sharpness, artifact models)
- Metric interfaces and tests
- Ready-to-run plotting examples

## Repository Contents

- `mri_recon/datasets/`: dataset interfaces and FastMRI loader
- `mri_recon/reconstruction/`: reconstruction methods
- `mri_recon/distortions/`: k-space/image distortion operators
- `mri_recon/metrics/`: metric interfaces and implementations
- `examples/`: runnable scripts for visual inspection
- `tests/`: unit tests

## Available Reconstructions

| Class | Category | Notes |
|---|---|---|
| `ZeroFilledReconstructor` | Classic | Inverse FFT baseline |
| `LandweberReconstructor` | Classic iterative | Gradient-descent style |
| `ConjugateGradientReconstructor` | Classic iterative | Normal-equation solver |
| `TikhonovReconstructor` | Classic regularized | Closed-form ridge in k-space |
| `POCSReconstructor` | Undersampled | Data consistency + thresholding |
| `FISTAL1Reconstructor` | Undersampled | Accelerated sparse reconstruction |
| `TVPDHGReconstructor` | Undersampled | TV-regularized primal-dual solver |
| `DeepInverseRAMReconstructor` | Deep learning wrapper | Optional DeepInverse model |

## Available Distortions

| Class | Group | Notes |
|---|---|---|
| `IsotropicResolutionReduction` | Resolution | Circular low-pass truncation |
| `AnisotropicResolutionChange` | Resolution | Axis-dependent low-pass |
| `ZeroFillDistortion` | Resolution | k-space zero-padding |
| `PhaseEncodeDecimation` | Resolution | Keep every R-th ky line |
| `VariableDensityBandwidthReduction` | Resolution | Smooth radial taper |
| `CoordinateScaling` | Resolution | k-space coordinate scaling |
| `Apodization` | Sharpness | Gaussian/Hamming/Hann/Kaiser window |
| `DirectionalSharpnessControl` | Sharpness | Axis-specific apodization |
| `HighFrequencyBoost` | Sharpness | Radial high-frequency gain |
| `UnsharpMaskKspace` | Sharpness | k-space unsharp masking |
| `RegularizedInverseBlur` | Sharpness | Wiener-like inverse filter |
| `GibbsRingingDistortion` | Artifacts | Hard rectangular truncation |
| `AliasingWrapAroundDistortion` | Artifacts | ky undersampling aliasing |
| `EPINHalfGhostDistortion` | Artifacts | Odd/even phase mismatch |
| `LineByLineMotionGhostDistortion` | Artifacts | Line-wise motion phase |
| `OffResonanceDistortion` | Artifacts | Per-line off-resonance phase accrual |

## Available Metrics

| Class | Type | Notes |
|---|---|---|
| `L1Metric` | Reference | Mean absolute error |
| `MSEMetric` | Reference | Mean squared error |
| `RMSEMetric` | Reference | Root mean squared error |
| `NMSEMetric` | Reference | Normalized MSE |
| `PSNRMetric` | Reference | Peak signal-to-noise ratio |
| `SSIMMetric` | Reference | Structural similarity |
| `UQIMetric` | Reference | Universal quality index |
| `GMSDMetric` | Reference | Gradient magnitude similarity deviation |
| `SREMetric` | Reference | Signal-to-reconstruction error |
| `LPIPSMetric` | Reference | Learned perceptual similarity |
| `EntropyMetric` | Non-reference | Intensity distribution entropy |
| `RMSContrastMetric` | Non-reference | RMS contrast |
| `TenengradMetric` | Non-reference | Gradient-based sharpness |
| `BlurEffectMetric` | Non-reference | Blur effect indicator |

## Setup

```bash
python -m pip install -r requirements.txt
```

For FastMRI usage, place/download data under:

- `data/fastmri/knee_singlecoil_val/singlecoil_val`
- or pass a custom `--source` path to examples

## Run Examples

From the repository root:

```bash
python examples/fastmri_reconstruction_plot.py --source data/fastmri/knee_singlecoil_val/singlecoil_val
```

```bash
python examples/fastmri_distortion_plot.py --algorithm zero-filled --source data/fastmri/knee_singlecoil_val/singlecoil_val
```

Outputs are saved in `reports/`.
