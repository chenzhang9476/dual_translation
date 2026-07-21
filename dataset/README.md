# Dataset

This directory contains both dataset code and the extraction location for the
two separate data archives.

| File | Purpose |
|---|---|
| `SynthRAD_HN.py` | Head-and-neck slice dataset with index loading, case filtering, foreground validation, and tensor loading. |
| `SynthRAD_AB.py` | Abdomen slice dataset. |
| `SynthRAD_TH.py` | Thorax slice dataset. |
| `split_data.py` | Dataset split generation helper. |
| `Task_split.py` | Task-aware split utility. |
| `evaluate_task.py` | Task-level image quality evaluation helper. |
| `__init__.py` | Public dataset imports. |

Expected extracted directories:

```text
synthRAD2025_Task1_Train/
synthRAD2025_Task1_Train_process/
synthRAD2025_Task2_Train/
synthRAD2025_Task2_Train_process/
```

Processed task directories must retain their `slice_index_*.json` and
`synthrad_split_*.json` files.
