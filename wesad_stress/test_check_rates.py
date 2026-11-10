from data import list_subjects, load_subject, check_rates
data_root = "../data/WESAD"  

subiecti = list_subjects(data_root)
print("Subiecți găsiți:", subiecti)

if subiecti:
    primul_subiect = subiecti[0]
    print(f"\nÎncarc datele pentru {primul_subiect}...")

    raw = load_subject(data_root, primul_subiect)

    rezultate = check_rates(raw)
    
    print("\nDuratele calculate pentru fiecare senzor (în secunde):")
    for senzor, durata in rezultate.items():
        print(f" - {senzor}: {durata:.2f} secunde")