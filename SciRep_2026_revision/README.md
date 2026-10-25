# Bio-inspired Amino Acid encoding for explainable IoT intrusion detection (CICIoT2023)

Code for the revised Scientific Reports paper. It rebuilds the data subset, the split, and all tables and figures in the paper.

Author: Thaer AL Ibaisi (De Montfort University)

DOI: https://doi.org/10.5281/zenodo.23641014

## Data
CICIoT2023 CSV release from https://www.unb.ca/cic/datasets/iotdataset-2023.html (one folder per class, 34 folders).

## Environment
Python 3.10 with the versions in requirements.txt. Biopython must be 1.77, because newer versions changed the secondary structure groups.

## How to run (in this order)
1. `python 01_sample.py <CSV folder> --balanced` : balanced 50k subset (25k benign, 25k attack, seed 42)
2. `python 03_select_k.py` : number of selected features (k) by 5-fold CV on the training set
3. `python 04_encode_folds.py` : Yeo-Johnson, selection and encoding, fitted per fold and for the final split
4. `python 05_models.py <model> ...` : 10-fold CV and test results (Tables 2, 3)
5. `python 06_ablation.py <A|B|C> <XGB|RNN>` : ablation (Table 5)
6. `python 07_analysis.py` : correlations, detection per family, RNN architecture, timings (Fig. 1, Tables 1, 4, 9)
7. `python 08a_xai_classical.py lime|shap|eli5` and `python 08b_xai_rnn.py shap|lime|captum|anchors|internals` : XAI figures and numbers
8. `python 09_extras.py` : worked example, family counts, correlation heatmap

`summarize.py` and `make_tables.py` print the results and the LaTeX table rows.

Models: KNN NB DT LR SVM RF GB XGB CatBoost Bagging Stacking-LR Stacking-DT CNN RNN LSTM GRU Transformer

The encoding is in pipeline.py (Yeo-Johnson, mutual information selection, ASCII mod 20, Amino Acid sequence, ten Biopython properties).

Scripts save their progress, so a stopped run can be started again and continues from where it stopped. The output of the run used in the paper is in results/.
