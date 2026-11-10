import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .protocols import get_splits

SENSOR_SETS = {
    "EDA": ["EDA"], "BVP": ["BVP"], "TEMP": ["TEMP"], "ACC": ["ACC"],
    "EDA+BVP": ["EDA", "BVP"], "EDA+BVP+TEMP": ["EDA", "BVP", "TEMP"],
    "ALL": ["EDA", "BVP", "TEMP", "ACC"],
}
META = {"subject", "seg", "seg_start", "seg_end", "t0", "t1", "y"}


# Filtrează tabelul mare și alege doar coloanele care corespund senzorilor pe care ai cerut să îi folosești
def feature_columns(df, sensors):
    prefixes = tuple(s + "_" for s in sensors)
    return [c for c in df.columns if c not in META and c.startswith(prefixes)]


def make_model(name, seed=0):
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    elif name == "rf":
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", # trateaza ambele clase cu importanta reala
                                     random_state=seed, n_jobs=-1)
    elif name == "svm":
        clf = SVC(kernel="rbf", class_weight="balanced")
    elif name == "majority":  # baseline: mereu clasa cea mai frecventă
        clf = DummyClassifier(strategy="most_frequent")
    elif name == "rf_shuffled":  # baseline: etichete de antrenare amestecate
        clf = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                     random_state=seed, n_jobs=-1)
    else:
        raise ValueError(name)
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), clf)


def evaluate(df, protocol, sensors_name, model_name, win_s, overlap, repeats=10, seed=0):
    cols = feature_columns(df, SENSOR_SETS[sensors_name])
    X = df[cols].values
    y = df["y"].values
    subj = df["subject"].values
    rows = []
    n_rep = repeats if protocol == "P1" else 1  # doar P1 este aleator
    for rep in range(n_rep):
        for tr, te in get_splits(df, protocol, win_s, seed=seed + rep):
            ytr = y[tr].copy()
            if model_name == "rf_shuffled":
                ytr = np.random.default_rng(seed + rep).permutation(ytr)
            model = make_model(model_name, seed + rep).fit(X[tr], ytr)
            pred = model.predict(X[te])
            for s in sorted(set(subj[te])):
                m = subj[te] == s
                yt, yp = y[te][m], pred[m]
                ba = balanced_accuracy_score(yt, yp) if len(set(yt)) > 1 else np.nan
                rows.append({"window_s": win_s, "overlap": overlap, "sensors": sensors_name,
                             "model": model_name, "protocol": protocol, "repeat": rep,
                             "subject": s, "balanced_acc": ba,
                             "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
                             "n_train": len(tr), "n_test": int(m.sum())})
    return rows


def summarize(results):
    keys = ["window_s", "overlap", "sensors", "model", "protocol"]
    per_subject = results.groupby(keys + ["subject"])[["balanced_acc", "f1_macro"]].mean()
    g = per_subject.groupby(keys)
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_sd"))
    return out.reset_index()


def protocol_gaps(summary):
    keys = ["window_s", "overlap", "sensors", "model"]
    wide = summary.pivot_table(index=keys, columns="protocol", values="balanced_acc_mean")
    for a, b in [("P1", "P2"), ("P2", "P3"), ("P1", "P3")]:
        if a in wide and b in wide:
            wide[f"{a}-{b}"] = wide[a] - wide[b]
    return wide.reset_index()
