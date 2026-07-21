# Baselines

This directory contains source-only copies of comparison methods:

- `cyclegan_pix2pix/`
- `readable_ddpm/`
- `conditional_ddpm/`

Embedded datasets, trained weights, result images, logs, caches, Git history,
and the second duplicate CycleGAN repository were intentionally excluded.
Each baseline keeps its upstream README and environment/config source. Dataset
paths and checkpoint paths must be configured for the new host before running.

These third-party implementations have their own CLIs and dependency
assumptions; they are not imported by the main DualTranslation entry points.
