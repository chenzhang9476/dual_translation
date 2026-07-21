# Core

`factory.py` defines `DualTranslationFactory`, the single construction path
for:

- selecting `SynthRAD_HN_Dataset`, `SynthRAD_AB_Dataset`, or
  `SynthRAD_TH_Dataset`;
- constructing the low-frequency and high-frequency UNets;
- constructing `DualDiffusion` with shared configuration fields;
- loading either a full checkpoint dictionary or a raw state dictionary.

`__init__.py` exposes the factory as the public API. Training and both
inference scripts import this module to avoid architecture drift.
