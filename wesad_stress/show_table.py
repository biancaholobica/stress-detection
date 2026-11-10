from data import list_subjects, load_subject
from features import extract_windows

# Încarcă datele subiectului S2
data_root = "../data/WESAD"  # Ajustează calea dacă e nevoie
raw = load_subject(data_root, "S2")

# Extrage ferestrele
df = extract_windows(raw, subject="S2", win_s=10, overlap=0.25)

# Afișează informații despre tabel
print("Dimensiune tabel:", df.shape)
print("\nPrimele 5 rânduri:")
print(df.head())

print("\nColoanele disponibile:")
print(df.columns.tolist())