"""
data_cleaner.py
---------------
Sva logika čišćenja i feature engineering-a za Airbnb dataset.
Poziva se NAKON data_loader.py, a PRIJE analize.

Pipeline po gradu:
  1. Čišćenje kolone 'price' ($1,234.00 → float)
  2. Filtriranje nultih/negativnih cijena
  3. Uklanjanje ekstremnih outliera (IQR × 1.5 po room_type grupi)
  4. Parsiranje 'amenities' JSON stringa u Python listu + count
  5. Konverzija boolean kolona ('t'/'f' → True/False)
  6. Računanje udaljenosti od centra grada (geodesic, km)
  7. Starost hosta u godinama (host_since → danas)
  8. Starost listinga u godinama (first_review → danas)
  9. Konverzija cijene u USD (price_usd) po aproksimativnom kursu
"""
import ast
import json
import numpy as np
import pandas as pd
from datetime import datetime
from geopy.distance import geodesic
from src.utils import CITY_CENTERS

# Referentni datum za izračun starosti hostova i listinga
REFERENCE_DATE = datetime(2026, 5, 21)

# Aproksimativni konverzijski kursevi u USD (maj 2026)
# 5 EUR gradova: Athens, Malaga, Naples, Rome, Valencia — kurs 1 EUR ≈ 1.08 USD
# Istanbul: 1 TRY ≈ 0.031 USD
CONVERSION_RATES = {
    'Athens':   lambda x: x * 1.08,
    'Malaga':   lambda x: x * 1.08,
    'Istanbul': lambda x: x * 0.031,
    'Naples':   lambda x: x * 1.08,
    'Rome':     lambda x: x * 1.08,
    'Valencia': lambda x: x * 1.08,
}

BOOLEAN_COLS = [
    'host_is_superhost',
    'instant_bookable',
    'has_availability',
    'host_has_profile_pic',
    'host_identity_verified',
]


# ---------------------------------------------------------------------------
# Pomoćne (private) funkcije
# ---------------------------------------------------------------------------

def _clean_price(value):
    """Konvertuje '$1,234.00' ili varijantu u float. Vraća NaN ako ne može."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace('$', '').replace(',', '').strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def _parse_amenities(value):
    """Parsira JSON/Python-lista string amenities u Python listu stringova."""
    if pd.isna(value):
        return []
    s = str(value)
    try:
        result = ast.literal_eval(s)
        return result if isinstance(result, list) else []
    except Exception:
        pass
    try:
        return json.loads(s.replace("'", '"'))
    except Exception:
        return []


def _remove_outliers_iqr(df, price_col='price', group_col='room_type'):
    """
    Uklanja gornje outliere koristeći IQR × 1.5 metodu.
    Grupira po room_type kako bi se različiti tipovi tretirali odvojeno.
    Donja granica je uvijek 0 (negativne cijene su već filtrirane).
    """
    if group_col not in df.columns or df.empty:
        return df

    valid_parts = []
    for _, group in df.groupby(group_col, observed=True):
        q1 = group[price_col].quantile(0.25)
        q3 = group[price_col].quantile(0.75)
        if pd.isna(q1) or pd.isna(q3):
            valid_parts.append(group)
            continue
        iqr = q3 - q1
        upper = q3 + 1.5 * iqr
        valid_parts.append(group[group[price_col] <= upper])

    return pd.concat(valid_parts, ignore_index=True) if valid_parts else df.iloc[0:0]


def _compute_distance(lat, lon, city_center):
    """Vraća geodesic udaljenost (km) od koordinata do centra grada."""
    try:
        if pd.notna(lat) and pd.notna(lon):
            return geodesic((lat, lon), city_center).km
    except Exception:
        pass
    return np.nan


# ---------------------------------------------------------------------------
# Javne funkcije
# ---------------------------------------------------------------------------

def clean_city(city, df):
    """
    Čisti i obogaćuje DataFrame za jedan grad.

    Parametri
    ---------
    city : str   — naziv grada (mora biti ključ u CITY_CENTERS)
    df   : pd.DataFrame — sirovi DataFrame iz data_loader

    Povratni
    --------
    pd.DataFrame — očišćeni i obogaćeni DataFrame
    """
    total_raw = len(df)
    print(f"\n  Čišćenje: {city} ({total_raw:,} sirovih redova)")

    # 1. Čišćenje kolone 'price'
    df['price'] = df['price'].apply(_clean_price)
    df = df[df['price'].notna() & (df['price'] > 0)].copy()
    after_price = len(df)
    print(f"     OK: Nakon čišćenja cijene:      {after_price:,} "
          f"(uklonjeno {total_raw - after_price:,})")

    # 2. Uklanjanje outliera po room_type grupama
    if 'room_type' in df.columns:
        df = _remove_outliers_iqr(df, 'price', 'room_type')
    after_outlier = len(df)
    print(f"     OK: Nakon uklanjanja outliera:  {after_outlier:,} "
          f"(uklonjeno {after_price - after_outlier:,})")

    # 3. Amenities — parsiranje i count
    if 'amenities' in df.columns:
        df['amenities_list'] = df['amenities'].apply(_parse_amenities)
        df['amenity_count']  = df['amenities_list'].apply(len)
    else:
        df['amenities_list'] = [[] for _ in range(len(df))]
        df['amenity_count']  = 0

    # 4. Boolean konverzija
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = df[col].map({'t': True, 'f': False})

    # 5. Udaljenost od centra (vectorized poziv)
    if city not in CITY_CENTERS:
        print(f"     Upozorenje: Centar za {city} nije definisan u CITY_CENTERS.")
        df['distance_from_center'] = np.nan
    else:
        center = CITY_CENTERS[city]
        df['distance_from_center'] = df.apply(
            lambda r: _compute_distance(r.get('latitude'), r.get('longitude'), center),
            axis=1
        )

    # 6. Starost hosta (godine)
    if 'host_since' in df.columns:
        df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
        df['host_age_years'] = (
            (REFERENCE_DATE - df['host_since']).dt.days / 365.25
        )

    # 7. Starost listinga (godine od prve recenzije)
    if 'first_review' in df.columns:
        df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
        df['listing_age_years'] = (
            (REFERENCE_DATE - df['first_review']).dt.days / 365.25
        )

    # 8. Konverzija cijene u USD
    converter = CONVERSION_RATES.get(city, lambda x: x)
    df['price_usd'] = converter(df['price'])

    # 9. Konzolni sažetak
    if not df.empty:
        avg_local = df['price'].mean()
        avg_usd   = df['price_usd'].mean()
        avg_dist  = df['distance_from_center'].mean()
        print(f"     OK: Prosječna cijena: {avg_local:.0f} lokalno | ~{avg_usd:.0f} USD")
        print(f"     OK: Prosj. udaljenost od centra: {avg_dist:.1f} km")
        print(f"     OK: Prosj. amenities po listingu: {df['amenity_count'].mean():.1f}")
    else:
        print(f"     Upozorenje: Nula redova ostalo nakon čišćenja za {city}!")

    return df


def clean_all_cities(raw_city_dfs):
    """
    Iterira kroz sve gradove i primjenjuje clean_city() na svaki.

    Parametri
    ---------
    raw_city_dfs : dict  {city_name: raw DataFrame}

    Povratni
    --------
    cleaned_dfs : dict  {city_name: cleaned DataFrame}
    combined    : pd.DataFrame — svi gradovi zajedno
    """
    cleaned_dfs = {}
    all_dfs = []

    for city, df in raw_city_dfs.items():
        try:
            cleaned = clean_city(city, df.copy())
            if not cleaned.empty:
                cleaned_dfs[city] = cleaned
                all_dfs.append(cleaned)
            else:
                print(f"     Upozorenje: {city} preskočen — prazan DataFrame nakon čišćenja.")
        except Exception as e:
            print(f"     Greška: Kritična greška pri čišćenju {city}: {e}")

    combined = (
        pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    )

    print(f"\n  Čišćenje završeno: {len(combined):,} listinga "
          f"kroz {len(cleaned_dfs)} gradova")
    return cleaned_dfs, combined
