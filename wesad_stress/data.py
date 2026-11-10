import pickle
from pathlib import Path

import numpy as np

FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
FS_LABEL = 700
KEEP = {1: 0, 3: 0, 2: 1}

#cauta in folderul de date subfoldere S<numar> care contin fisierul S<numar>.pk1 si le sorteaza dupa numar
def list_subjects(root):
    found = [p.name for p in Path(root).glob("S*") if (p / f"{p.name}.pkl").exists()]
    return sorted(found, key=lambda s: int(s[1:]))


# deschide pickle ul unui subiect si returneaza un dictionar cu BVP, EDA, TEMP, ACC si etichetele
def load_subject(root, subject):
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
    out = {k: len(raw[k]) / FS[k] for k in FS}
    out["label"] = len(raw["label"]) / FS_LABEL
    return out


# separa etichetele in segmente de aceeasi stare 
def label_segments(label):
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
