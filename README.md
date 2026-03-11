# mri_recon

Minimal Python package for MRI reconstruction utilities.

## Dataset interfaces

The repository now includes a compact `mri_recon.datasets` package with:

- `BaseDataset`: abstract dataset interface with shared `download` and
  `apply_normalization` helpers.
- `FastMRIDataset`: a FastMRI-specific implementation that reads real
  fastMRI-style HDF5 volumes from `<root>/<split>/<sample_id>.h5`, exposing
  slice-wise `kspace`, optional `mask`, and optional reconstruction targets.

Install the required runtime dependency before using `FastMRIDataset`:

```bash
python -m pip install -r requirements.txt
```

Example:

```python
from mri_recon.datasets import FastMRIDataset

dataset = FastMRIDataset("/path/to/data", split="train", challenge="multicoil")
sample = dataset.read_sample("file1000001", slice_index=0)
normalized = dataset.apply_normalization(sample, field="target", method="minmax")
```
