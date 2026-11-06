"""Teste care verifică exact ce contează pentru corectitudine: ferestre curate și lipsa scurgerii de date."""
import numpy as np
import pytest

from make_synthetic import make_subject
from wesad_stress.data import KEEP, label_segments
from wesad_stress.features import extract_windows
from wesad_stress.protocols import split_p1, split_p2, split_p3

import pandas as pd


@pytest.fixture(scope="module")
def table():
    rng = np.random.default_rng(1)
    frames = []
    for s in ["S2", "S3", "S4"]:
        raw = make_subject(rng)
        raw = {"BVP": raw["signal"]["wrist"]["BVP"].ravel(), "EDA": raw["signal"]["wrist"]["EDA"].ravel(),
               "TEMP": raw["signal"]["wrist"]["TEMP"].ravel(), "ACC": raw["signal"]["wrist"]["ACC"],
               "label": raw["label"]}
        frames.append(extract_windows(raw, s, win_s=30, overlap=0.5))
    return pd.concat(frames, ignore_index=True)


def test_labels_binary_and_segments_clean():
    label = np.array([0] * 700 + [1] * 1400 + [4] * 700 + [2] * 1400)
    segs = label_segments(label)
    assert [s[2] for s in segs] == [1, 2]            # 0 și 4 sunt excluse
    assert set(KEEP.values()) == {0, 1}


def test_windows_inside_segments(table):
    assert (table["t0"] >= table["seg_start"] - 1e-9).all()
    assert (table["t1"] <= table["seg_end"] + 1e-9).all()
    assert (table["t1"] - table["t0"]).round(6).eq(30).all()


def test_p2_has_gap_and_no_overlap(table):
    (tr, te), = split_p2(table, win_s=30)
    assert len(tr) > 0 and len(te) > 0
    assert len(set(tr) & set(te)) == 0
    for _, g in table.groupby(["subject", "seg"]):
        pos = table.index.get_indexer(g.index)
        g_tr = g[np.isin(pos, tr)]
        g_te = g[np.isin(pos, te)]
        if len(g_tr) and len(g_te):
            assert g_te["t0"].min() - g_tr["t1"].max() >= 30 - 1e-6  # pauză ≥ lungimea ferestrei


def test_p3_subjects_disjoint(table):
    for tr, te in split_p3(table):
        assert set(table["subject"].iloc[tr]).isdisjoint(set(table["subject"].iloc[te]))


def test_p1_mixes_neighbours(table):
    """P1 NU separă ferestrele vecine: cel puțin o fereastră de test are un vecin în antrenare."""
    (tr, te), = split_p1(table, seed=0)
    tr_set = set(tr)
    has_neighbour = any((i - 1) in tr_set or (i + 1) in tr_set for i in te)
    assert has_neighbour
