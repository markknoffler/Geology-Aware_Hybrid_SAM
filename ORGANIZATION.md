# SAM Directory Organization

This folder is now separated into two main areas:

- `codebase/` - source code, scripts, notebooks, and runnable project files
- `resources/` - datasets, result artifacts, documents, and static assets

## Structure

- `codebase/`
  - model and training/inference Python files
  - `segment_anything/` package and related build metadata
  - `scripts/`, `demo/`, `notebooks/`, and example files
- `resources/`
  - `data/` - datasets and generated cache-like non-code folders
  - `results/` - metrics and experiment output folders
  - `docs/` - paper and project documentation files
  - `assets/` - architecture/image assets

## Note

If you run scripts from inside `codebase/`, relative imports and paths remain simpler.