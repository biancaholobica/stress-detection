"""Rulează experimentele. Exemple:

  python run_experiment.py --data-root data/WESAD --quick
  python run_experiment.py --data-root data/WESAD --windows 5 10 30 60 120 --overlaps 0 0.5
  python run_experiment.py --data-root data/synthetic --quick     (test fără date reale)
"""
import argparse
import time
from pathlib import Path

import pandas as pd

from wesad_stress.data import check_rates, list_subjects, load_subject
from wesad_stress.experiment import SENSOR_SETS, evaluate, protocol_gaps, summarize
from wesad_stress.features import extract_windows


def build_table(root, win_s, overlap, cache_dir):
    cache = Path(cache_dir) / f"{Path(root).name}_w{win_s}_o{overlap}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    frames = []
    for s in list_subjects(root):
        raw = load_subject(root, s)
        frames.append(extract_windows(raw, s, win_s, overlap))
    df = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--windows", type=int, nargs="+", default=[60])
    ap.add_argument("--overlaps", type=float, nargs="+", default=[0.0, 0.5])
    ap.add_argument("--sensors", nargs="+", default=["EDA", "ALL"], choices=list(SENSOR_SETS))
    ap.add_argument("--models", nargs="+", default=["logreg", "rf"])
    ap.add_argument("--protocols", nargs="+", default=["P1", "P2", "P3"])
    ap.add_argument("--repeats", type=int, default=10, help="repetări pentru P1")
    ap.add_argument("--controls", action="store_true", help="adaugă majority și rf_shuffled")
    ap.add_argument("--quick", action="store_true", help="configurație mică, pentru prima rulare")
    ap.add_argument("--check-rates", action="store_true", help="afișează duratele și iese")
    ap.add_argument("--out", default="results/results.csv")
    ap.add_argument("--cache", default="cache")
    a = ap.parse_args()

    if a.check_rates:
        s = list_subjects(a.data_root)[0]
        print(s, {k: round(v, 1) for k, v in check_rates(load_subject(a.data_root, s)).items()})
        return
    if a.quick:
        a.windows, a.sensors, a.models, a.repeats = [60], ["EDA", "ALL"], ["logreg", "rf"], 5
    models = a.models + (["majority", "rf_shuffled"] if a.controls else [])

    results = []
    t0 = time.time()
    for w in a.windows:
        for ov in a.overlaps:
            df = build_table(a.data_root, w, ov, a.cache)
            print(f"fereastră {w}s, suprapunere {ov}: {len(df)} ferestre, "
                  f"{df['subject'].nunique()} subiecți, stres={df['y'].mean():.2f}")
            for sens in a.sensors:
                for m in models:
                    for p in a.protocols:
                        results += evaluate(df, p, sens, m, w, ov, repeats=a.repeats)
    res = pd.DataFrame(results)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False)
    summ = summarize(res)
    summ.to_csv(Path(a.out).with_name("summary.csv"), index=False)
    gaps = protocol_gaps(summ)
    gaps.to_csv(Path(a.out).with_name("gaps.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print("\nDiferențe între protocoale (balanced accuracy):")
    print(gaps.round(3).to_string(index=False))
    print(f"\ntimp: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
