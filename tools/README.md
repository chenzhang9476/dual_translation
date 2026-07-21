# Tools

- `preprocessing/`: upstream SynthRAD preprocessing stages and configs.
- `verify_release.py`: validates the code, checkpoint, and optional data
  layout after extraction.
- `generate_file_index.py`: regenerates `FILE_INDEX.md` from the release
  contents.

Run:

```bash
python tools/verify_release.py --require-data --require-checkpoints
python tools/generate_file_index.py
```
