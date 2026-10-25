#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:04:38 2026

@author: thaer

Proposed pipeline (as in our IoT paper): Yeo-Johnson transformation -> feature
selection (mutual information, top-k) -> Amino Acid encoding -> ten structural
properties. All steps are fitted on training data only.
"""
import numpy as np, pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.feature_selection import mutual_info_classif
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AA = "ARNDCEQGHILKMFPSTWYV"
PROPS = ["MolecularWeight", "Aromaticity", "InstabilityIndex", "IsoelectricPoint",
         "AlphaHelix", "ReducedCysteines", "DisulfideBridges", "Gravy", "BetaTurn", "BetaStrand"]

def to_sequence(values):
    """Each value -> rounded to 2 decimals, plain decimal string; the strings are joined
    in feature order; each character -> AA[ASCII mod 20]."""
    s = "".join(format(round(float(v), 2), ".2f") for v in values)
    return "".join(AA[ord(c) % 20] for c in s)

def properties(seq):
    X = ProteinAnalysis(seq)
    helix, turn, sheet = X.secondary_structure_fraction()   # Biopython 1.77: (VIYFWL, NPGS, EMAL)
    red, ss = X.molar_extinction_coefficient()
    return [X.molecular_weight(), X.aromaticity(), X.instability_index(), X.isoelectric_point(),
            helix, red, ss, X.gravy(), turn, sheet]

class AminoAcidEncoder:
    def __init__(self, k, seed=42):
        self.k, self.seed = k, seed
    def fit(self, X, y):
        self.yj = PowerTransformer(method="yeo-johnson", standardize=True).fit(X)
        Z = self.yj.transform(X)
        self.mi = mutual_info_classif(Z, y, random_state=self.seed)
        self.selected = np.argsort(-self.mi)[: self.k]
        return self
    def sequences(self, X):
        Z = self.yj.transform(X)[:, self.selected]
        return [to_sequence(r) for r in Z]
    def transform(self, X):
        return np.array([properties(s) for s in self.sequences(X)])
