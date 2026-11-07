# Compute reference mass, ion and attention features using inherited equations.
# 按既有公式计算质量、离子和 attention 特征。
"""Retained pi-NovoQC feature calculations used by the CTC adapter.

SPDX-License-Identifier: MIT

The feature functions below are retained from
``pi-NovoQC-main/cct_qc_graphnovo/FDR.py`` in the user's official ZIP.
PTM242 tokens missing from Pyteomics are added to ``mass.std_aa_mass`` before
feature extraction, so the official ``get_theo_mz`` function is unchanged.

No sequencing runner or FDR scorer is embedded in this feature module.
"""

import re
from collections import Counter

import numpy as np
from scipy.stats import entropy, spearmanr

from pyteomics import mass


# basic useful dictionary -- official pi-NovoQC FDR.py
ion_type_dict = {
    "0": "a",
    "1": "b",
    "2": "y",
    "3": "noise",
}
stat_type_dict = {
    "0": "1_",
    "1": "1_NH3",
    "2": "1_H2O",
    "3": "2_",
    "4": "2_NH3",
    "5": "2_H2O",
}


def get_theo_mz(peptide):
    """
    ion_type_dict = {
        '0': 'a',
        '1': 'b',
        '2': 'y',
    }
    stat_type_dict = {
        '0': '1_',
        '1': '1_NH3',
        '2': '1_H2O',
        '3': '1_H2O_NH3',
        '4': '1_H2O_H2O',
        '5': '2_',
        '6': '2_NH3',
        '7': '2_H2O',
        '8': '2_H2O_NH3',
        '9': '2_H2O_H2O',
    }
    """
    # Define charge states
    charges = [1, 2]
    # Generate b-ions with NH3 and H2O loss for each charge state
    a_ions = []
    b_ions = []
    y_ions = []
    peptide = re.split(r"(?<=.)(?=[A-Z])", peptide)
    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[:i], ion_type='a', charge=charge)
            a_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            a_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            a_ions.append(ion_mass_h2o)

    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[:i], ion_type='b', charge=charge)
            b_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            b_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            b_ions.append(ion_mass_h2o)

        # Generate y-ions with NH3 and H2O loss for each charge state
    for i in range(1, len(peptide)):
        for charge in charges:
            ion_mass = mass.fast_mass(peptide[-i:], ion_type='y', charge=charge)
            y_ions.append(ion_mass)
            ion_mass_nh3 = ion_mass - mass.Composition('NH3').mass()
            y_ions.append(ion_mass_nh3)
            ion_mass_h2o = ion_mass - mass.Composition('H2O').mass()
            y_ions.append(ion_mass_h2o)
    return np.array([a_ions, b_ions, y_ions])


def _install_ptm242_masses(residue_masses):
    """Add only token labels absent from the official Pyteomics mass table."""

    added = []
    for token, token_mass in residue_masses.items():
        if token in mass.std_aa_mass:
            continue
        mass.std_aa_mass[token] = float(token_mass)
        added.append(token)
    return added


def get_aby_labels(seqs, exp_mz, theo_mz):
    aby_label = []
    for i in range(len(exp_mz)):
        if np.min(abs(theo_mz - exp_mz[i])) <= 0.05:
            ion_type_idx, stat_type_idx = np.where(abs(theo_mz - exp_mz[i])==np.min(abs(theo_mz - exp_mz[i])))
            peak_ion_type = ion_type_dict[str(ion_type_idx[0])]
            ion_idx, stat_idx = int(stat_type_idx[0]/len(stat_type_dict))+1, stat_type_idx[0]%len(stat_type_dict)
            peak_stat_type = stat_type_dict[str(stat_idx)]
            aby_label.append(f'{peak_ion_type}{ion_idx}-{peak_stat_type}')
        else:
            aby_label.append('N')
    return aby_label


class PeptideMass:
    """A simple class for calculating peptide masses

    Parameters
    ----------
    residues: Dict or str {"massivekb", "canonical"}, optional
        The amino acid dictionary and their masses. By default this is only
        the 20 canonical amino acids, with cysteine carbamidomethylated. If
        "massivekb", this dictionary will include the modifications found in
        MassIVE-KB. Additionally, a dictionary can be used to specify a custom
        collection of amino acids and masses.
    """

    canonical = {
        "G": 57.021463735,
        "A": 71.037113805,
        "S": 87.032028435,
        "P": 97.052763875,
        "V": 99.068413945,
        "T": 101.047678505,
        "C+57.021": 103.009184505 + 57.02146,
        "L": 113.084064015,
        "I": 113.084064015,
        "N": 114.042927470,
        "D": 115.026943065,
        "Q": 128.058577540,
        "K": 128.094963050,
        "E": 129.042593135,
        "M": 131.040484645,
        "H": 137.058911875,
        "F": 147.068413945,
        # "U": 150.953633405,
        "R": 156.101111050,
        "Y": 163.063328575,
        "W": 186.079312980,
        # "O": 237.147726925,
    }

    # Modfications found in MassIVE-KB
    massivekb = {
        # N-terminal mods:
        "+42.011": 42.010565,  # Acetylation
        "+43.006": 43.005814,  # Carbamylation
        "-17.027": -17.026549,  # NH3 loss
        "+43.006-17.027": (43.006814 - 17.026549),
        # AA mods:
        "M+15.995": canonical["M"] + 15.994915,  # Met Oxidation
        "N+0.984": canonical["N"] + 0.984016,  # Asn Deamidation
        "Q+0.984": canonical["Q"] + 0.984016,  # Gln Deamidation
    }

    # Constants
    hydrogen = 1.007825035
    oxygen = 15.99491463
    h2o = 2 * hydrogen + oxygen
    proton = 1.00727646688

    def __init__(self, residues="canonical"):
        """Initialize the PeptideMass object"""
        if residues == "canonical":
            self.masses = self.canonical
        elif residues == "massivekb":
            self.masses = self.canonical
            self.masses.update(self.massivekb)
        else:
            self.masses = residues

    def __len__(self):
        """Return the length of the residue dictionary"""
        return len(self.masses)

    def mass(self, seq, charge=None):
        """Calculate a peptide's mass or m/z.

        Parameters
        ----------
        seq : list or str
            The peptide sequence, using tokens defined in ``self.residues``.
        charge : int, optional
            The charge used to compute m/z. Otherwise the neutral peptide mass
            is calculated

        Returns
        -------
        float
            The computed mass or m/z.
        """
        if isinstance(seq, str):
            seq = re.split(r"(?<=.)(?=[A-Z])", seq)

        calc_mass = sum([self.masses[aa] for aa in seq]) + self.h2o
        if charge is not None:
            calc_mass = (calc_mass / charge) + self.proton

        return calc_mass


def compute_fused_score(attn_score, intensity_rank, alpha=0.3, beta=0.7):
    eps = 1e-8
    return alpha * attn_score + beta * intensity_rank


def get_rank_percentile(intensity_list):
    sorted_vals = np.sort(intensity_list)[::-1]
    total = len(sorted_vals)
    rank_map = {v: i for i, v in enumerate(sorted_vals)}
    return [1.0 - rank_map[val] / total for val in intensity_list]


def compute_aa_confidences(attention_matrix, theor_frags, pred_pep, mz, intens, top_k=5):
    """
    attention_matrix: Rows from top to bottom correspond to the peptide sequence from right to left (C‑terminus to N‑terminus).
    """
    """
    Calculate the confidence score for each amino acid in the sequence and mark anchor positions (high attention with matched fragment peaks).
    - peptide_sequence: Peptide sequence string.
    - spectrum_peaks: [(mz, intensity), ...] List of spectral peaks.
    - cross_attention: 2D array or tensor of shape [L, M] representing the attention distribution from each sequence position to the spectral input features.
    - tolerance_ppm: Mass tolerance (in ppm) for theoretical peak matching.
    - top_k: For each amino acid, examine the top‑k attention peaks for checking.

    Returns: (aa_confidences, aa_details)
        aa_confidences: A list of length L, each element is the confidence score (0~1) of the corresponding amino acid.
        aa_details: A list of length L, each element is a dictionary containing information about that amino acid site (index, aa, is_anchor, confidence, matched_peak_mz, matched_peak_intensity, relative_intensity, attention_score).
    """
    pred_pep_reversed = list(reversed(re.split(r"(?<=.)(?=[A-Z])", pred_pep)))
    L = len(pred_pep_reversed)
    aa_confidences = [0.0] * L
    aa_details = []
    intens_rank = get_rank_percentile(intens)
    for i in range(attention_matrix.shape[0]): 
        top_k_indices = np.argpartition(attention_matrix[i], -top_k)[-top_k:]
        aa_info = {
            "index": i,
            "aa": pred_pep_reversed[i],
            'theor_frag_y': f'y{i+1}',
            'theor_frag_b': f'b{len(pred_pep_reversed)-(i+1+1)}',
            "is_anchor": False,
            "confidence": 0.0,
            "matched_peak_mz": None,
            "matched_peak_intensity": None,
            "relative_intensity": None,
            "attention_score": None
        }
        confidence_score = 0.0
        best_match_score = 0.0
        for idx in top_k_indices:
            peak_mz = mz[idx]
            peak_intensity = intens[idx]
            peak_intensity_rank = intens_rank[idx]
            attn_weight = attention_matrix[i][idx]     
            match_b = False
            match_y = False
            theor_b_mz = None
            theor_y_mz = None
            if theor_frags[idx] != 'N':
                if 'y' in theor_frags[idx]:
                    if abs(int(theor_frags[idx].split('-')[0][1:])-i) <= 2:
                        match_y = True
                if 'b' in theor_frags[idx]:
                    if abs(int(theor_frags[idx].split('-')[0][1:]) - (len(pred_pep_reversed)-(i+1))) <= 2:
                        match_b = True
            if match_b or match_y:
                if aa_info["matched_peak_mz"] is None:
                    aa_info["matched_peak_mz"] = peak_mz
                    aa_info["matched_peak_intensity"] = peak_intensity
                    aa_info["intensity_rank"] = peak_intensity_rank
                    aa_info["attention_score"] = attn_weight
                score = compute_fused_score(attn_weight, peak_intensity_rank)
                confidence_score += score       
                aa_info["is_anchor"] = True
        aa_conf = confidence_score
        aa_info["confidence"] = aa_conf
        aa_confidences[i] = aa_conf
        aa_details.append(aa_info)
    return aa_confidences, aa_details


def compute_global_metrics(peptide_sequence, theor_frags, pred_pep, mz, intens):
    """
    Calculate peptide‑level quality control metrics: fragment coverage, number of complementary ion pairs, and explained intensity.
    Return a dictionary {coverage, complement_fraction, explained_intensity}.
    """
    pred_pep_reversed = list(reversed(re.split(r"(?<=.)(?=[A-Z])", pred_pep)))
    seq_length = len(pred_pep_reversed)
    total_theoretical = len(theor_frags)
    num_matched = sum(np.array(theor_frags) != 'N')
    coverage = num_matched / total_theoretical if total_theoretical > 0 else 0.0
    total_intensity = sum(intens)
    explained_intensity = sum(intens[np.array(theor_frags) != 'N']) / total_intensity
    
    return {
        "coverage": coverage,
        "explained_intensity": explained_intensity
    }


def evaluate_peptide_sequence(attention_matrix, theor_frags, peptide_sequence, mz, intens, top_k=3, global_factor=0.1,
                              masses=None, precursor_mass=None, precursor_charge=None, precursor_tol_ppm=20, 
                              rt=None, rt_diff_threshold=None):
    """
    Evaluate the matching quality between a given peptide sequence and the spectrum, returning a result dictionary containing multi-level quality control information.

    Parameters:
      - peptide_sequence (str): The de novo predicted peptide sequence.
      - spectrum_peaks (list of (mz, intensity)): List of spectrum peak tuples.
      - cross_attention (ndarray or tensor): Transformer cross-attention matrix with shape [L, M].
      - precursor_mass (float): Experimental monoisotopic precursor m/z (optional).
      - precursor_tol_ppm (float): Precursor mass tolerance in ppm, used for mass deviation judgment.
      - rt (float): Experimental retention time of the spectrum (optional).
      - rt_diff_threshold (float): Retention time tolerance threshold (optional, if predicted RT model results are available).

    Returns:
      A dictionary containing quality control assessments at both the amino acid and peptide levels, suitable for JSON serialization.
    """
    # 1. Calculate confidence scores and anchor information for each amino acid
    aa_confidences, aa_details = compute_aa_confidences(attention_matrix, theor_frags, peptide_sequence, mz, intens, top_k)
    # 2. Calculate global quality control metrics and preliminary global confidence score
    metrics = compute_global_metrics(peptide_sequence, theor_frags, peptide_sequence, mz, intens)
    coverage = metrics["coverage"]
    explained_intensity = metrics["explained_intensity"]
    global_confidence = (coverage + explained_intensity) / 2.0
    metrics["global_confidence_initial"] = global_confidence
    # 3. Check precursor mass match and penalize the confidence score if it exceeds the threshold
    mass_pass = True
    if precursor_mass is not None:
        predicted_mz = masses.mass(peptide_sequence, precursor_charge)
        
        mass_error = 0.0
        if predicted_mz != 0:
            mass_error = abs(predicted_mz - precursor_mass) / predicted_mz * 1e6  
        if mass_error > precursor_tol_ppm:
            mass_pass = False
            global_confidence = global_confidence * global_factor
            metrics["mass_error_ppm"] = mass_error
        else:
            metrics["mass_error_ppm"] = mass_error
    result = {
        "sequence": peptide_sequence,
        "aa_confidences": np.array(aa_confidences),
#         "aa_details": aa_details,
        "metrics": metrics,
        "global_confidence": global_confidence,
        "mass_pass": mass_pass
    }
    if rt is not None:
        result["rt"] = rt
        result["rt_pass"] = rt_pass
    return result


def compute_attention_features(attn_matrix, theor_mz_list, mz_array, intensity_array, attention_threshold=0.05):
    """
    Calculate a set of attention features at both the amino acid and peptide levels.

    Parameters:
        attn_matrix: np.ndarray of shape [L_aa, L_ms], the attention from each amino acid in the protein sequence to each spectral peak.
        theor_mz_list: list of theoretical fragment m/z values, length L_aa.
        mz_array: array of experimental m/z values, shape [L_ms].
        intensity_array: array of intensity values, shape [L_ms].
        attention_threshold: float, threshold for determining significant attention spectral peaks.

    Returns:
        features: dict, containing all statistical features at both the amino acid and peptide levels.
    """
    L_aa, L_ms = attn_matrix.shape
    features = {
        "aa_entropy": [],
        "aa_concentration": [],
        "aa_offset": [],
        "aa_center_drift": [],
        "aa_span": [],
    }

    intensity_array = np.array(intensity_array)
    mz_array = np.array(mz_array)

    aa_peak_indices = []  

    for i in range(L_aa):
        attn_row = attn_matrix[i]
        attn_row = attn_row / (attn_row.sum() + 1e-8)  

        # entropy
        ent = entropy(attn_row)
        features["aa_entropy"].append(ent)

        # Concentration Ratio (Maximum Attention Proportion)
        max_attn = attn_row.max()
        features["aa_concentration"].append(max_attn)

        # Matching Offset (m/z difference between the maximum peak and theoretical fragment m/z)
        peak_idx = np.argmax(attn_row)
        aa_peak_indices.append(peak_idx)
        offset = abs(mz_array[peak_idx] - theor_mz_list[i]) if theor_mz_list[i] is not None else None
        features["aa_offset"].append(offset)

        # Center Shift (Difference between Weighted Center and Theoretical m/z)
        weighted_center = np.sum(attn_row * mz_array)
        center_drift = abs(weighted_center - theor_mz_list[i]) if theor_mz_list[i] is not None else None
        features["aa_center_drift"].append(center_drift)

        # Span (Standard Deviation)
        std_dev = np.sqrt(np.sum(attn_row * (mz_array - weighted_center)**2))
        features["aa_span"].append(std_dev)

    # peptide-level feature
    attn_sim = []
    for i in range(L_aa - 1):
        sim = np.dot(attn_matrix[i], attn_matrix[i+1]) / (
            np.linalg.norm(attn_matrix[i]) * np.linalg.norm(attn_matrix[i+1]) + 1e-8)
        attn_sim.append(sim)
    features["avg_attention_similarity"] = np.mean(attn_sim)
    peak_mzs = mz_array[aa_peak_indices]
    rho, _ = spearmanr(list(range(L_aa)), peak_mzs)
    features["fragment_alignment_consistency"] = rho
    return features


def compute_psm_features_with_attention(evaluate_results, attn_matrix, theor_mz_list, mz_array, intensity_array):
    """
    Parameters:
        evaluate_results: Output from the original evaluate_peptide_sequence function
        attn_matrix: Cross-attention matrix [L_aa, L_ms]
        theor_mz_list: list, m/z of theoretical fragments for each amino acid
        mz_array: Experimental spectrum m/z array
        intensity_array: Experimental spectrum intensity array

    Returns:
        feature_dict: dict, containing all merged features
    """
    base_features = {}
    aa_conf_list = evaluate_results['aa_confidences']
    # 1. List of amino acid confidence scores (e.g., with length equal to the peptide length)
    base_features["aa_conf_mean"] = sum(aa_conf_list) / len(aa_conf_list)
    base_features["aa_conf_var"] = np.var(aa_conf_list)
    base_features["aa_conf_min"] = min(aa_conf_list)
    base_features["aa_conf_max"] = max(aa_conf_list)
    base_features['num_low_conf'] = sum([1 for c in aa_conf_list if c < 0.2])
    base_features['num_high_conf'] = sum([1 for c in aa_conf_list if c > 0.8])
    # 2. Global quality control score
    base_features["global_qc_score"] = evaluate_results['global_confidence']
    # 3. Fragment coverage calculation
    base_features["fragment_coverage"] = evaluate_results['metrics']['coverage']
    # 4. Explained intensity percentage
    base_features["explained_intensity"] = evaluate_results['metrics']['explained_intensity']
    # 5. Precursor mass deviation (ppm)  (not for DIA)
    base_features["precursor_error_ppm"] = evaluate_results['metrics']['mass_error_ppm']
    # Anchor statistics (temporarily using high confidence as a proxy for anchor points)
    base_features['anchor_ratio'] = base_features['num_high_conf'] / len(aa_conf_list)

    # Add attention-derived features
    attn_features = compute_attention_features(
        attn_matrix, theor_mz_list, mz_array, intensity_array
    )

    full_features = {**base_features}
    for k, v in attn_features.items():
        if isinstance(v, list):
            full_features[f"{k}_mean"] = np.mean(v)
            full_features[f"{k}_std"] = np.std(v)
            full_features[f"{k}_min"] = np.min(v)
            full_features[f"{k}_max"] = np.max(v)
        else:
            full_features[k] = v
    return full_features


