import numpy as np
from sklearn.model_selection import train_test_split


# P1 - Random Split
# Ia toate rândurile din tabel, le amestecă la întâmplare și le 
# împarte în două grupuri: 75% pentru antrenament (tr) și 25% pentru testare (te).
def split_p1(df, seed, test_size=0.25):
    idx = np.arange(len(df))
    tr, te = train_test_split(idx, test_size=test_size, stratify=df["y"].values,
                              random_state=seed)
    return [(tr, te)]


# P2 - Temporal Split
# Antrenare pe prima parte și testare pe a doua parte a fiecărei etape 
# experimentale, cu un interval de timp între ele
def split_p2(df, win_s, test_frac=0.25):
    tr_pos, te_pos = [], []
    for _, g in df.groupby(["subject", "seg"]):
        s, e = g["seg_start"].iloc[0], g["seg_end"].iloc[0]
        cut = s + (1.0 - test_frac) * (e - s)
        te_pos.extend(df.index.get_indexer(g.index[g["t0"].values >= cut - 1e-9]))
        tr_pos.extend(df.index.get_indexer(g.index[g["t1"].values <= cut - win_s + 1e-9]))
    return [(np.array(tr_pos), np.array(te_pos))]


# P3 - LOSO
# Antrenare pe 14 subiecți, testare pe al 15-lea subiect
def split_p3(df):
    subj = df["subject"].values
    folds = []
    for s in sorted(set(subj)):
        folds.append((np.flatnonzero(subj != s), np.flatnonzero(subj == s)))
    return folds


def get_splits(df, protocol, win_s, seed=0):
    if protocol == "P1":
        return split_p1(df, seed)
    if protocol == "P2":
        return split_p2(df, win_s)
    if protocol == "P3":
        return split_p3(df)
    raise ValueError(f"protocol necunoscut: {protocol}")
