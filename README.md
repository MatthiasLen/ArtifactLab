# mri_recon

Minimal Python package for MRI reconstruction utilities.

## Dataset interfaces

The repository now includes a compact `mri_recon.datasets` package with:

- `BaseDataset`: abstract dataset interface with shared `download` and
  `apply_normalization` helpers.
- `FastMRIDataset`: a wrapper around the real `fastmri.data.SliceDataset`
  class that reads FastMRI HDF5 volumes slice-by-slice and exposes convenient
  helpers for downloading a sample, sampling slices, and converting matrices
  to NumPy.

Install the real fastMRI dependency stack before using `FastMRIDataset`:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -r requirements.txt
```

Example:

```python
from mri_recon.datasets import FastMRIDataset

dataset = FastMRIDataset(split="val", challenge="singlecoil")
dataset.download(source="/path/to/fastmri_or_archive")

sample = dataset[0]
pixel_matrix = dataset.to_numpy(sample, field="target")
```

## Reconstruction interfaces

The package also exposes a compact `mri_recon.reconstruction` module with:

- `BaseReconstructor`: abstract interface exposing `apply_reconstruction`.
- `ZeroFilledReconstructor`: centered inverse FFT baseline for Cartesian MRI.
- `LandweberReconstructor`: small iterative least-squares reconstructor.
- `TVPDHGReconstructor`: total-variation reconstruction using a primal-dual
  hybrid gradient solver for single-coil undersampled k-space.
- `DeepInverseReconstructor`: optional wrapper around DeepInverse `RAM`,
  `VarNet`, `MoDL`, and `deepinv.optim.optim_builder`.

Example:

```python
from mri_recon.datasets import FastMRIDataset
from mri_recon.reconstruction import ZeroFilledReconstructor

dataset = FastMRIDataset(split="val", challenge="singlecoil")
dataset.download()

sample = dataset[0]
image = ZeroFilledReconstructor().apply_reconstruction(sample)
```

To use `DeepInverseReconstructor`, install the optional DeepInverse dependency
from `requirements.txt`. The wrapper can build a simple MRI physics operator
directly from a FastMRI sample and also exposes the most relevant documented
pretrained DeepInverse models via `available_pretrained_models()`:

```python
from mri_recon.datasets import FastMRIDataset
from mri_recon.reconstruction import DeepInverseReconstructor

dataset = FastMRIDataset(split="val", challenge="singlecoil")
dataset.download()
sample = dataset[0]

physics = DeepInverseReconstructor.build_mri_physics(sample)
reconstructor = DeepInverseReconstructor("varnet", physics=physics)
reconstruction = reconstructor.apply_reconstruction(sample)
magnitude_image = reconstructor.to_magnitude_image(reconstruction)
```

The wrapper exposes these documented pretrained DeepInverse models/backbones:

```python
DeepInverseReconstructor.available_pretrained_models()
# ('ram', 'drunet', 'dncnn')
```

For the direct pretrained reconstructor:

```python
ram_model = DeepInverseReconstructor.load_pretrained_model("ram")
```

Run the plotting example directly on local knee single-coil test data:

```bash
python examples/fastmri_reconstruction_plot.py --source data/fastmri/knee_singlecoil_test/singlecoil_test
```

For less aggressive TV regularization on near fully sampled single-coil data,
use a smaller TV weight and fewer PDHG iterations:

```bash
python examples/fastmri_reconstruction_plot.py --source data/fastmri/knee_singlecoil_test/singlecoil_test --tv-weight 2e-4 --tv-iterations 60
```
