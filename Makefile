CONFIG=configs/default.yaml

prepare:
	python scripts/01_prepare_encoded_dataset.py --config $(CONFIG)

train-classical:
	python scripts/02_train_classical_holdout.py --config $(CONFIG)

train-deep:
	python scripts/03_train_deep_holdout.py --config $(CONFIG)

cv:
	python scripts/04_cross_validate_deep.py --config $(CONFIG)

xai-classical:
	python scripts/05_xai_baselines.py --config $(CONFIG)

xai-rnn:
	python scripts/06_xai_rnn.py --config $(CONFIG)

bench:
	python scripts/07_benchmark_models.py --config $(CONFIG)

tables:
	python scripts/08_build_paper_tables.py --config $(CONFIG)
