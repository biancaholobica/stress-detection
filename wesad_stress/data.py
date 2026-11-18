"""Încărcarea setului WESAD (fișierele S<id>/S<id>.pkl) și segmentarea etichetelor.

Constantele de mai jos sunt cele din readme-ul WESAD, după cum le știu eu.
Verifică-le cu check_rates() pe datele reale (vezi README).
"""
import pickle
from pathlib import Path

import numpy as np

# frecvențe de eșantionare (Hz) ale senzorilor de la încheietură (Empatica E4)
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
FS_LABEL = 700  # etichetele sunt date la frecvența senzorilor de la piept

# 1 = relaxare (baseline), 2 = stres, 3 = amuzament; 0, 4, 5, 6, 7 se exclud
# clasa binară: stres = 1, non-stres (relaxare + amuzament) = 0
KEEP = {1: 0, 3: 0, 2: 1}


def list_subjects(root):
    """Găsește folderele S<id> care conțin fișierul S<id>.pkl."""
    found = [p.name for p in Path(root).glob("S*") if (p / f"{p.name}.pkl").exists()]
    return sorted(found, key=lambda s: int(s[1:]))


def load_subject(root, subject):
    """Citește un subiect și returnează doar semnalele de la încheietură + etichetele."""
    path = Path(root) / subject / f"{subject}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1") 
    wrist = d["signal"]["wrist"]
    return {
        "BVP": np.asarray(wrist["BVP"]).ravel(),
        "EDA": np.asarray(wrist["EDA"]).ravel(),
        "TEMP": np.asarray(wrist["TEMP"]).ravel(),
        "ACC": np.asarray(wrist["ACC"]),
        "label": np.asarray(d["label"]).ravel(),
    }


def check_rates(raw):
    """Durata în secunde dedusă din fiecare semnal; valorile trebuie să fie apropiate.

    Dacă diferă mult, frecvențele din FS nu corespund fișierelor tale.
    """
    out = {k: len(raw[k]) / FS[k] for k in FS}
    out["label"] = len(raw["label"]) / FS_LABEL
    return out


def label_segments(label):
    """Împarte vectorul de etichete în segmente continue cu aceeași etichetă.

    Returnează o listă de (start_s, end_s, eticheta) doar pentru etichetele păstrate.
    """
    label = np.asarray(label)
    change = np.flatnonzero(np.diff(label)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(label)]
    segs = []
    for s, e in zip(starts, ends):
        lab = int(label[s])
        if lab in KEEP:
            segs.append((s / FS_LABEL, e / FS_LABEL, lab))
    return segs


def validate_raw(raw, tol=0.02):
    """Oprește rularea dacă datele nu se potrivesc cu constantele din cod.

    Verifică (1) că duratele deduse din semnale coincid cu cea a etichetelor și
    (2) că etichetele 1, 2 și 3 există. Așa o frecvență greșită nu mai produce
    în tăcere rezultate greșite.
    """
    d = check_rates(raw)
    ref = d["label"]
    bad = {k: round(v, 1) for k, v in d.items() if abs(v - ref) / ref > tol}
    if bad:
        raise ValueError(f"Duratele nu coincid ({bad}, față de label={ref:.1f}s): "
                         "verifică FS și FS_LABEL din data.py")
    codes = set(np.unique(raw["label"]).astype(int).tolist())
    if not {1, 2, 3} <= codes:
        raise ValueError(f"Lipsesc etichetele 1, 2 sau 3; găsite: {sorted(codes)}")
    return d