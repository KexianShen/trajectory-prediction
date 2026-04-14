# Repository Guidelines

## Project Structure & Module Organization

Core code lives in `src/`. Use `src/model/` for Lightning trainers and model entry points, `src/model/layers/` for attention and encoder blocks, `src/datamodule/` for AV2 extraction and dataloading, `src/metrics/` for validation metrics, and `src/utils/` for optimizers, callbacks, and submission helpers. Hydra config lives in `conf/`, especially `conf/model/*.yaml`, `conf/datamodule/av2.yaml`, and `conf/config.yaml`. Top-level scripts are `train.py`, `eval.py`, and `preprocess.py`.

## Build, Test, and Development Commands

Install dependencies with:

```bash
pip install -r requirements.txt
```

Prepare cached AV2 features after setting `data_root` in `conf/config.yaml`:

```bash
python preprocess.py
```

Train the main model or pretraining variants:

```bash
python train.py monitor=val_AvgMinFDE
python train.py model=model_mtm epochs=20
python train.py model=model_mrm epochs=20
```

Run evaluation after setting `checkpoint` in `conf/config.yaml`:

```bash
python eval.py test=true
```

Use `python train.py fast_dev_run=1` for a quick trainer smoke test. For syntax checks, run `python -m py_compile` on edited modules.

## Coding Style & Naming Conventions

Follow standard Python style: 4-space indentation, `snake_case` for functions and variables, `PascalCase` for classes, and concise type hints where already used. Keep reusable attention logic in `src/model/layers/attn.py`; do not reintroduce a separate `rope_attn.py`. Prefer explicit keyword arguments when wiring Hydra-instantiated trainers and model blocks.

When changing shared model infrastructure, prefer adapting the corresponding implementation from the local `flow` repo at `/mnt/disk/repo/mac/flow` instead of writing a fresh version in this project. This applies especially to trainers, encoders/decoders, attention blocks, optimizer wiring, and Hydra config structure. Keep AV2-specific data processing, metrics, and submission logic local to this repo unless there is a clear `flow` counterpart.

## Testing Guidelines

There is no dedicated `tests/` suite yet. Treat syntax checks and small model smoke tests as the baseline:

```bash
python -m py_compile src/model/model.py src/model/layers/attn.py
```

When changing training or data code, run at least one short Hydra job such as `python train.py fast_dev_run=1`. When touching attention, verify tensor shapes and mask flow through `TempoNet`, `SpaNet`, `InterNet`, and submission formatting.

## Commit & Pull Request Guidelines

Recent commits use short, imperative subjects such as `remove duplicate code` and `maximize tensor core usage`. Keep commit titles brief, lowercase is acceptable, and focus on one change per commit. PRs should state the affected training phase (`all`, `mtm`, or `mrm`), summarize config changes, list validation steps run, and include sample metrics or logs when behavior changes.

## Configuration Tips

Do not hardcode dataset or checkpoint paths. Route new options through `conf/` and trainer/model constructors, following patterns like `headwise_attn_output_gate`, `preprocess_num_cpus`, and `preprocess_batch_size`.
