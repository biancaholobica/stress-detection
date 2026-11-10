import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, find_peaks

from .data import FS, KEEP, label_segments


# pastreaza doar frecventele utile( de ex bataile inimii) folosit pentru puls
# elimina zgomotul lent si cel rapid
def _bandpass(x, fs, lo, hi, order=3):
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

# lasa sa treaca doar variatiile lente - folosit pentru EDA
def _lowpass(x, fs, cutoff, order=2):
    sos = butter(order, cutoff, btype="low", fs=fs, output="sos")
    return sosfiltfilt(sos, x)

# ia datele brute de la toti senzorii si le pregateste o singura data pentru toata inregistrarea
# EDA_PH - vârfurile rapide de transpirație care apar instant când omul are o emoție sau un stres.
def prepare_signals(raw):
    eda = raw["EDA"].astype(float)
    tonic = _lowpass(eda, FS["EDA"], 0.05)
    return {
        "BVP": _bandpass(raw["BVP"].astype(float), FS["BVP"], 0.5, 8.0),
        "EDA": eda,
        "EDA_PH": eda - tonic,
        "TEMP": raw["TEMP"].astype(float),
        "ACC": np.linalg.norm(raw["ACC"].astype(float), axis=1),
    }


# pentru o fereastra de timp dintr un senzor calculeaza:
# media(mean), abaterea standard(std), min, max
# panta 
def _slope(x, fs):
    t = np.arange(len(x)) / fs
    t = t - t.mean()
    return float((t * (x - x.mean())).sum() / (t * t).sum())


def _stats(x, fs, name, with_slope=True):
    out = {f"{name}_mean": x.mean(), f"{name}_std": x.std(),
           f"{name}_min": x.min(), f"{name}_max": x.max()}
    if with_slope:
        out[f"{name}_slope"] = _slope(x, fs)
    return out


# functie speciala pentru inima
# calculeaza: BVP hr(ritmul cardiac mediu), BVP_sdnn(cat de neregulate sunt intervalele dintre batai), 
# BVP_rmssd(scade puternic atunci când omul este stresat)
def _bvp_features(bvp):
    fs = FS["BVP"]
    nan = {"BVP_hr": np.nan, "BVP_sdnn": np.nan, "BVP_rmssd": np.nan}
    sd = bvp.std()
    if sd == 0:
        return nan
    peaks, _ = find_peaks(bvp, distance=int(0.4 * fs), prominence=0.3 * sd)
    ibi = np.diff(peaks) / fs * 1000.0  # ms
    if len(ibi) < 3:
        return nan
    return {"BVP_hr": 60000.0 / ibi.mean(), "BVP_sdnn": ibi.std(),
            "BVP_rmssd": float(np.sqrt(np.mean(np.diff(ibi) ** 2)))}


# ia o parte mică de timp (de la secunda t0 la t1, de exemplu o fereastră de 10 secunde) și calculează
# toate statisticiile, numără vârfurile de transpirație (EDA_n_peaks) și calculează pulsul pentru acea bucățică.
def window_features(sig, t0, t1):
    def cut(name):
        fs = FS[name if name in FS else "EDA"]
        i0 = int(round(t0 * fs))
        i1 = int(round(t1 * fs))
        return sig[name][i0:i1], fs

    eda, fs_e = cut("EDA")
    ph, _ = cut("EDA_PH")
    bvp, _ = cut("BVP")
    temp, fs_t = cut("TEMP")
    acc, fs_a = cut("ACC")

    feats = {}
    feats.update(_stats(eda, fs_e, "EDA"))
    peaks, _ = find_peaks(ph, prominence=0.01)
    feats["EDA_n_peaks"] = len(peaks)
    feats.update(_bvp_features(bvp))
    feats.update(_stats(temp, fs_t, "TEMP"))
    feats.update(_stats(acc, fs_a, "ACC", with_slope=False))
    return feats


# folosește segmentele curate găsite la label segments (ca să nu treacă niciodată granița dintre starea de stres și cea de
# relaxare) și plimbă fereastra pas cu pas pe tot parcursul experimentului (folosind un overlap / suprapunere, de ex. 25%).
# suprapunerea are loc doar in interiorul aceleiasi stari!
def extract_windows(raw, subject, win_s, overlap):
    sig = prepare_signals(raw)
    step = win_s * (1.0 - overlap)
    rows = []
    for seg_id, (s, e, lab) in enumerate(label_segments(raw["label"])):
        t = s
        while t + win_s <= e + 1e-9:
            row = {"subject": subject, "seg": seg_id, "seg_start": s, "seg_end": e,
                   "t0": t, "t1": t + win_s, "y": KEEP[lab]}
            row.update(window_features(sig, t, t + win_s))
            rows.append(row)
            t += step
    return pd.DataFrame(rows)
