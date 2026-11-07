# Check worker merges, unknown labels and cohort filtering.
# 核对分片合并、未知标签以及整批过滤的行为。
import numpy as np
import pandas as pd
import pytest
from MNovo.inference.multi_gpu import batch_shards, selected_devices, merge_predictions
from MNovo.cct.cct_pipeline import validate_roles, known_rows
from MNovo.cct.cct_scoring import select_psms


def test_whole_batches_preserve_batch_context():
    ids = np.arange(35)[::-1]
    shards = batch_shards(ids, 8, 3)
    batches = [tuple(ids[i:i+8]) for i in range(0, len(ids), 8)]
    actual = [tuple(s[i:i+8]) for s in shards for i in range(0, len(s), 8)]
    assert sorted(batches) == sorted(actual)
    assert sorted(np.concatenate(shards)) == sorted(ids)


def test_visible_device_mapping():
    assert selected_devices("1,0", "GPU-a,GPU-b") == ["GPU-b", "GPU-a"]
    with pytest.raises(ValueError):
        selected_devices("0,0")
    with pytest.raises(ValueError):
        selected_devices("2", "0,1")


def test_merge_restores_order_and_rejects_duplicates(tmp_path):
    a, b = tmp_path / "a.tsv", tmp_path / "b.tsv"
    a.write_text("Sequence\tdataset_index\nAK\t2\n")
    b.write_text("Sequence\tdataset_index\nKA\t1\n")
    output = tmp_path / "merged.tsv"
    merge_predictions([a, b], [1, 2], output)
    assert output.read_text() == "Sequence\tdataset_index\nKA\t1\nAK\t2\n"
    with pytest.raises(ValueError):
        merge_predictions([a, a], [2], output)


def test_physical_groups_not_peptide_keys_define_roles():
    a = pd.DataFrame(dict(pair_id=[1, 1], group_id=["scan1"]*2, peptide_key=["AK"]*2))
    b = a.assign(pair_id=2, group_id="scan2")
    validate_roles(dict(train=a, calibration=b))
    with pytest.raises(ValueError):
        validate_roles(dict(train=a, calibration=b.assign(group_id="scan1")))


def test_unknown_not_negative_supervision():
    assert known_rows(pd.DataFrame(dict(denovo_label=[0, 1, np.nan]))).denovo_label.tolist() == [0, 1]


def test_floor_does_not_remove_high_score_targets():
    rows = pd.DataFrame(dict(is_decoy=[0, 1, 0, 1], score=[.99, .8, .7, .6]))
    target, _, cutoff = select_psms(rows, .05, .01, require_truth=False, score_bounds=(.75, .85))
    assert .99 in target.score.tolist()
    assert cutoff >= .75


def test_spearman_only_repair_policy():
    from MNovo.cct.cct_feature_policy import self_test
    assert self_test()["status"] == "PASS"


def test_unknown_reference_not_negative_and_modifications_preserved():
    from MNovo.cct.cct_runtime import CCTFeatureExporter
    from types import SimpleNamespace
    exporter = CCTFeatureExporter.__new__(CCTFeatureExporter)
    predictions = [SimpleNamespace(peptide="AM+15.995IK")] * 5
    assert exporter._labels(predictions, [None, np.nan, "", "AM+15.995LK", "AMIK"]) == [None, None, None, 1, 0]


def test_mgf_title_cannot_become_reference(tmp_path):
    from MNovo.input import materialize_mgf_lmdb
    from MNovo.denovo.spectrum_index import LmdbSpectrumIndex
    mgf = tmp_path / "mixed.mgf"
    mgf.write_text("".join("BEGIN IONS\nTITLE=AK\nPEPMASS=351\nCHARGE=2+\n" + seq +
                          "100 20\n200 30\n300 50\nEND IONS\n" for seq in ("SEQ=AGK\n", "")))
    db = tmp_path / "mixed.lmdb"
    materialize_mgf_lmdb([mgf], db, 10, annotated=False)
    index = LmdbSpectrumIndex(str(db), None, lock=False)
    try:
        assert index.reference_sequence(0) == "AGK"
        assert index.reference_sequence(1) is None
        assert index[0][-1] == "AK"
    finally:
        index.env.close()
