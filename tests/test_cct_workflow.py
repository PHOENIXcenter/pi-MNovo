from pathlib import Path
from types import SimpleNamespace
import json
import numpy as np
import pandas as pd
import pytest
from MNovo.cct import workflow as wf


@pytest.mark.parametrize("table,extra", [
    ("TITLE\tSEQ\nr.1.1.2\tAM+15.995K", {}),
    ("File_Name\tSequence\tModification\nr.1.1.2\tAMK\t2,Oxidation[M];", {}),
    ("Raw file\tScan number\tCharge\tModified sequence\nr\t1\t2\t_AM(ox)K_", {}),
])
def test_join_accepted_labels(tmp_path, table, extra):
    p = tmp_path/"r.mgf"
    original = "".join(f"BEGIN IONS\nTITLE=r.{i}.{i}.2\nCHARGE=2+\nEND IONS\n" for i in (1,2))
    p.write_text(original)
    t = tmp_path/"accepted.tsv"
    t.write_text(table)
    cfg = wf.fit_settings(dict(reference_table=str(t), **extra))
    dest, = wf.annotate_sources([p], cfg, tmp_path/"labelled", ["A","M+15.995","K"])
    assert dest.read_text().count("SEQ=AM+15.995K") == 1
    assert p.read_text() == original and dest.read_text().count("END IONS") == 2


def test_automatic_workflow(tmp_path, monkeypatch):
    rng = np.random.default_rng(9)
    n = 256
    class Lookup:
        def __init__(self, _): pass
        def __enter__(self): return self
        def __exit__(self, *_): pass
        def spectrum(self, i):
            return dict(title=f"run.{i//2}.{i//2}.2", mz_array=[i], intensity_array=[1])
    monkeypatch.setattr(wf, "LmdbTitleLookup", Lookup)
    def predict(args, *_):
        assert args.fdr_mode == "off"
        Path(args.output).write_text("native")
        path = Path(args.cct_features_output)
        rows = [dict(pair_id=i, sequence=f"PEPTIDE{i}", valid=True, is_decoy=s, denovo_label=i%2,
                     **{f"f_{j:02}":rng.normal() for j in range(33)}) for i in range(n) for s in (0,1)]
        pd.DataFrame(rows).to_csv(path,index=False)
        Path(str(path)+".receipt.json").write_text(json.dumps(dict(schema=wf.pipeline.SCHEMA,
            scope_id="study",runtime_identity={"fixture":True},features_sha256=wf.pipeline._sha256(path))))
        return n
    args = SimpleNamespace(output=str(tmp_path/"native.tsv"), cct_profile=None, cct_features_output=None,
                           fdr_mode="on", cct_scope_id="study", cct_fit=wf.fit_settings(dict(epochs=1,gamma_method="suffix")),
                           psm_fdr=1., peptide_fdr=1.)
    assert wf.run_workflow(args,None,[],n,predict) == n
    manifest = pd.read_csv(tmp_path/"native.cct-work/roles.tsv", sep="\t")
    assert manifest.groupby(manifest.pair_id//2).role.nunique().eq(1).all()
    table = pd.read_csv(tmp_path/"native.cct/summary.tsv",sep="\t")
    assert list(table.Scope) == ["Held-out test"]*2 + ["Full application"]*2
    accepted = pd.read_csv(tmp_path/"native.cct/accepted_peptides.tsv", sep="\t")
    assert 0 < len(accepted) <= n
    assert table.Retained.iloc[-1] == len(accepted)
    assert (tmp_path/"native.cct-work/fitted/cct_profile.pt").is_file()


def test_no_truth_keeps_predictions(tmp_path):
    output = tmp_path/"native.tsv"
    args = SimpleNamespace(output=str(output), cct_features_output=None, cct_profile=None, cct_fit=wf.fit_settings(None))
    def predict(opts,*_):
        output.write_text("native")
        Path(opts.cct_features_output).write_text("pair_id,valid,is_decoy,denovo_label\n0,True,0,\n0,True,1,\n")
        return 1
    with pytest.raises(RuntimeError,match="No usable reference-labelled predictions"):
        wf.run_workflow(args,None,[],1,predict)
    assert output.read_text() == "native" and not output.with_suffix(".cct").exists()
