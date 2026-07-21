# SynthRAD Preprocessing

This folder preserves the original preprocessing implementation without raw
or generated data.

| File | Purpose |
|---|---|
| `stage1.py` | First preprocessing stage driven by `stage1_config.csv`. |
| `stage2.py` | Second preprocessing stage driven by `stage2_config.csv`. |
| `preprocessing_utils.py` | Shared image registration, resampling, masking, and I/O helpers. |
| `analysis_paper.py` | Dataset/preprocessing analysis helper. |
| `stage1_config.csv` | Stage 1 configuration. |
| `stage2_config.csv` | Stage 2 configuration. |
| `configs/` | Additional upstream preprocessing configs. |
| `docker/` | Upstream preprocessing container files. |
| `utilities/` | Auxiliary upstream utilities. |
| `UPSTREAM_README.md` | Original detailed preprocessing instructions. |

Review and adapt all input/output paths in the CSV config files before
reprocessing. The packaged processed-data archives can be used directly when
reprocessing is not required.
