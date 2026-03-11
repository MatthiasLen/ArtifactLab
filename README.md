# mri_recon

Minimal Python package for MRI reconstruction utilities.

## Dataset interfaces

The repository now includes a compact `mri_recon.datasets` package with:

- `BaseDataset`: abstract dataset interface with shared `download` and
  `apply_normalization` helpers.
- `FastMRIDataset`: a FastMRI-specific implementation that reads sample files
  from `<root>/<split>/<sample_id>.json`.

Example:

```python
from mri_recon.datasets import FastMRIDataset

dataset = FastMRIDataset("/path/to/data", split="train")
sample = dataset.read_sample("sample_001")
normalized = dataset.apply_normalization(sample)
```
