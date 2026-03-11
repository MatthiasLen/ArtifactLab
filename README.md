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
dataset.download()  # uses the built-in sample source when no official data is present

sample = dataset[0]
pixel_matrix = dataset.to_numpy(sample, field="target")
```
