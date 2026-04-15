from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

AMINO_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class EncodedRecord:
    amino_sequence: str
    molecular_weight: float
    aromaticity: float
    instability_index: float
    isoelectric_point: float
    alpha_helix: float
    reduced_cysteines: float
    disulfide_bridges: float
    gravy: float
    beta_turn: float
    beta_strand: float


def canonicalize_value(value: object) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (float, int)):
        return f"{float(value):.6g}"
    return str(value)


def row_to_ascii(row: pd.Series, feature_columns: Iterable[str]) -> str:
    parts = [f"{col}={canonicalize_value(row[col])}" for col in feature_columns]
    return "|".join(parts)


def ascii_to_amino(text: str) -> str:
    if not text:
        return "A"
    residues = []
    for ch in text:
        residues.append(AMINO_ALPHABET[ord(ch) % len(AMINO_ALPHABET)])
    return "".join(residues)


def compute_structural_properties(amino_sequence: str) -> EncodedRecord:
    if not amino_sequence:
        amino_sequence = "A"
    analyzer = ProteinAnalysis(amino_sequence)
    mw = float(analyzer.molecular_weight())
    aromaticity = float(analyzer.aromaticity())
    instability = float(analyzer.instability_index())
    pI = float(analyzer.isoelectric_point())
    helix, turn, sheet = analyzer.secondary_structure_fraction()
    ext_red, ext_disulf = analyzer.molar_extinction_coefficient()
    gravy = float(analyzer.gravy())
    return EncodedRecord(
        amino_sequence=amino_sequence,
        molecular_weight=mw,
        aromaticity=float(aromaticity),
        instability_index=float(instability),
        isoelectric_point=float(pI),
        alpha_helix=float(helix),
        reduced_cysteines=float(ext_red),
        disulfide_bridges=float(ext_disulf),
        gravy=float(gravy),
        beta_turn=float(turn),
        beta_strand=float(sheet),
    )


def encode_dataframe(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    records = []
    for _, row in df.iterrows():
        ascii_repr = row_to_ascii(row, feature_columns)
        aa = ascii_to_amino(ascii_repr)
        props = compute_structural_properties(aa)
        records.append(
            {
                "AminoSequence": props.amino_sequence,
                "MolecularWeight": props.molecular_weight,
                "Aromaticity": props.aromaticity,
                "InstabilityIndex": props.instability_index,
                "IsoelectricPoint": props.isoelectric_point,
                "AlphaHelix": props.alpha_helix,
                "ReducedCysteines": props.reduced_cysteines,
                "DisulfideBridges": props.disulfide_bridges,
                "Gravy": props.gravy,
                "BetaTurn": props.beta_turn,
                "BetaStrand": props.beta_strand,
            }
        )
    return pd.DataFrame.from_records(records)
