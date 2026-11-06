import argparse
import pickle
from pathlib import Path

import numpy as np

# (etichetă, durată în secunde); 0 = tranziție, 4 = meditație (se exclud din experiment)
TIMELINE = [(0, 20), (1, 240), (0, 15), (2, 180), (0, 15), (3, 120), (4, 120)]
HR = {0: 70, 1: 66, 2: 88, 3: 72, 4: 62}  # puls (bătăi/min) pe stare


def make_subject(rng):
    dur = sum(d for _, d in TIMELINE)
    lab_s = np.concatenate([[l] * d for l, d in TIMELINE])  # câte o etichetă pe secundă
    offset = rng.normal(0, 6)  # diferență între persoane
    label = np.repeat(lab_s, 700)

    # BVP la 64 Hz: undă de puls cu frecvență dependentă de stare
    hr_s = np.array([HR[l] + offset for l in lab_s])
    f = np.repeat(hr_s / 60.0, 64)
    phase = np.cumsum(2 * np.pi * f / 64)
    bvp = np.sin(phase) + 0.3 * np.sin(2 * phase) + rng.normal(0, 0.2, len(phase))

    # EDA la 4 Hz: nivel de bază + reacții rapide, mai frecvente în stres
    n = dur * 4
    lev = rng.uniform(2, 6) + np.repeat(np.where(lab_s == 2, 1.5, 0.0), 4)
    rate = np.repeat(np.where(lab_s == 2, 6.0, 2.0) / 60.0 / 4.0, 4)
    events = (rng.random(n) < rate).astype(float)
    kernel = np.exp(-np.arange(0, 40) / 8.0)
    eda = lev + 0.6 * np.convolve(events, kernel)[:n] + rng.normal(0, 0.01, n)

    temp = 33 + rng.normal(0, 0.8) + np.cumsum(rng.normal(0, 0.002, n))
    acc_sd = np.repeat(np.where(lab_s == 2, 0.25, 0.1), 32)  # vorbitul în stres
    acc = rng.normal(0, 1, (dur * 32, 3)) * acc_sd[:, None] + np.array([0, 0, 1.0])
    return {"signal": {"wrist": {"BVP": bvp[:, None], "EDA": eda[:, None],
                                 "TEMP": temp[:, None], "ACC": acc}},
            "label": label, "subject": "synthetic"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--subjects", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    for i in range(a.subjects):
        sid = f"S{i + 2}"
        p = Path(a.out) / sid
        p.mkdir(parents=True, exist_ok=True)
        with open(p / f"{sid}.pkl", "wb") as f:
            pickle.dump(make_subject(rng), f)
    print(f"{a.subjects} subiecți sintetici în {a.out}")
