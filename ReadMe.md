# Explainable IoT Intrusion Detection via Bio-Inspired Encoding

This repository has experiments pipeline described in the attached IEEE draft:
**"Explainable IoT Intrusion Detection via Bio-Inspired Encoding and Interpretable AI Models"**.

It covers:

- CICIoT-style CSV ingestion
- stratified 50,000-sample subset creation
- deterministic ASCII → amino-acid encoding
- computation of 10 structural properties
- classical ML benchmarking
- deep learning benchmarking (CNN, RNN, LSTM, GRU, Transformer, optional GNN)
- holdout and cross-validation evaluation
- XAI generation (LIME, SHAP, ELI5, Captum, Alibi)
- latency / throughput / parameter benchmarking
- table export for manuscript integration

## Important note

To reproduce the paper numbers as closely as possible, you still need to align:

- the exact CICIoT2023 subset
- preprocessing decisions
- random seeds
- exact train/test split
- exact model hyperparameters
- hardware / software versions

## Repository layout

```text
configs/default.yaml
scripts/01_prepare_encoded_dataset.py
scripts/02_train_classical_holdout.py
scripts/03_train_deep_holdout.py
scripts/04_cross_validate_deep.py
scripts/05_xai_baselines.py
scripts/06_xai_rnn.py
scripts/07_benchmark_models.py
scripts/08_build_paper_tables.py
src/iotxai/
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Edit `configs/default.yaml`, then run:

```bash
python scripts/01_prepare_encoded_dataset.py --config configs/default.yaml
python scripts/02_train_classical_holdout.py --config configs/default.yaml
python scripts/03_train_deep_holdout.py --config configs/default.yaml
python scripts/04_cross_validate_deep.py --config configs/default.yaml
python scripts/05_xai_baselines.py --config configs/default.yaml
python scripts/06_xai_rnn.py --config configs/default.yaml
python scripts/07_benchmark_models.py --config configs/default.yaml
python scripts/08_build_paper_tables.py --config configs/default.yaml
```

## Expected inputs

Point `data.raw_path` in the config to:

- a single CSV file, or
- a directory containing many CSV files from CICIoT2023

The loader infers numeric feature columns and the label column (default: `label`).

## Generated outputs

The pipeline writes into `outputs/`:

- `encoded_dataset.csv`
- `metrics_holdout_classical.json`
- `metrics_holdout_deep.json`
- `metrics_cv_deep.json`
- `tables/*.csv`
- `figures/*.png`
- saved models under `outputs/models/`

## Mapping and structural properties

Each row is serialized deterministically to a compact ASCII string, then each character is mapped into the 20 amino-acid alphabet using a vigesimal-style modulo mapping.
The following 10 structural features are computed:

1. MolecularWeight
2. Aromaticity
3. InstabilityIndex
4. IsoelectricPoint
5. AlphaHelix
6. ReducedCysteines
7. DisulfideBridges
8. Gravy
9. BetaTurn
10. BetaStrand

These are implemented with `Bio.SeqUtils.ProtParam.ProteinAnalysis`, with extinction coefficients used for reduced/disulfide features.
