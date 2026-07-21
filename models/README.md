# Models

| File | Purpose |
|---|---|
| `unet_simple.py` | Low-frequency conditional UNet operating on the approximation band. |
| `unet_complex.py` | High-frequency conditional UNet operating on detail bands. |
| `diffusion_dual.py` | Wavelet decomposition, low/high diffusion processes, combined training loss, sampling, and visualization helpers. |
| `__init__.py` | Stable imports for the model package. |

Model construction should go through `core.DualTranslationFactory` unless a
baseline experiment explicitly needs a different architecture.
