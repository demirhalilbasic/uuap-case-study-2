"""
data_loader.py
--------------
Isključivo odgovoran za čitanje sirovih CSV fajlova s diska.
Nikakvo čišćenje, nikakva transformacija — samo I/O operacije.
Čišćenje i feature engineering se nalaze u data_cleaner.py.
"""
import os
import pandas as pd
from src.utils import CITY_FILES


def load_all_cities(data_path='data/listings'):
    """
    Učitava sirove CSV fajlove za svaki grad iz CITY_FILES rječnika.

    Returns
    -------
    city_dfs : dict  {city_name: raw DataFrame}
    combined : pd.DataFrame  — sve gradove konkatenisane (sirovo)
    """
    city_dfs = {}
    all_dfs = []

    for city, file_prefix in CITY_FILES.items():
        print(f"  Učitavanje: {city}...")
        file_path = os.path.join(data_path, f"{file_prefix}.csv")

        if not os.path.exists(file_path):
            print(f"     Upozorenje: Fajl nije pronađen: {file_path}. Preskakanje...")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"     Greška pri čitanju {file_path}: {e}")
            continue

        if 'price' not in df.columns:
            print(f"     Greška: Nedostaje kolona 'price' za {city}. Preskakanje...")
            continue

        # Jedina transformacija koja se radi ovdje: dodavanje identifikatora grada
        df['city'] = city

        null_pct = df.isnull().mean().mean() * 100
        print(f"     OK: {len(df):,} redova | {len(df.columns)} kolona | "
              f"Prosječno null: {null_pct:.1f}%")

        city_dfs[city] = df
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    print(f"\n  Učitavanje završeno: {len(combined):,} sirovih listinga "
          f"iz {len(city_dfs)} gradova")
    return city_dfs, combined
