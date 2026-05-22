import os
import pandas as pd
import numpy as np
import ast
import json
from datetime import datetime
from geopy.distance import geodesic
from src.utils import CITY_CENTERS, CITY_FILES

def clean_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, str):
        price_str = price_str.replace('$', '').replace(',', '')
    try:
        return float(price_str)
    except ValueError:
        return np.nan

def parse_amenities(amenities_str):
    if pd.isna(amenities_str):
        return []
    try:
        return ast.literal_eval(amenities_str)
    except:
        try:
            return json.loads(amenities_str.replace("'", '"'))
        except:
            return []

def load_all_cities(data_path='data/listings'):
    city_dfs = {}
    all_dfs = []

    # Referentni privremeni datum za starosnu statistiku listinga i hostova (izvještaj Spring 2026)
    current_date = datetime(2026, 5, 21)

    conversion_rates = {
        'Athens': lambda x: x * 1.08,
        'Malaga': lambda x: x * 1.08,
        'Istanbul': lambda x: x * 0.031,
        'Naples': lambda x: x * 1.08,
        'Rome': lambda x: x * 1.08,
        'Valencia': lambda x: x * 1.08
    }

    boolean_cols = ['host_is_superhost', 'instant_bookable', 'has_availability',
                    'host_has_profile_pic', 'host_identity_verified']

    for city, file_prefix in CITY_FILES.items():
        print(f"Učitavanje podataka za grad: {city}...")
        file_path = os.path.join(data_path, f"{file_prefix}.csv")

        if not os.path.exists(file_path):
            print(f"    Upozorenje: Datoteka {file_path} nije pronađena. Preskakanje proceduri...")
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"    Sistemska greška pri čitanju {file_path}: {e}")
            continue

        total_rows = len(df)
        print(f"    - Inicijalni broj opservacija: {total_rows}")

        # 1. Čišćenje price
        if 'price' in df.columns:
            df['price'] = df['price'].apply(clean_price)
            df = df[(df['price'].isna()) | (df['price'] > 0)]
        else:
            print("    Upozorenje: Nedostaje fundamentalna kolona 'price'.")
            continue

        clean_price_rows = len(df)
        print(f"    - Opservacije nakon generalnog filtriranja cijena: {clean_price_rows}")

        # 2. Čišćenje amenities
        if 'amenities' in df.columns:
            df['amenities_list'] = df['amenities'].apply(parse_amenities)
            df['amenity_count'] = df['amenities_list'].apply(len)
        else:
            df['amenities_list'] = [[] for _ in range(len(df))]
            df['amenity_count'] = 0

        # 3. Konverzija booleana
        for col in boolean_cols:
            if col in df.columns:
                df[col] = df[col].map({'t': True, 'f': False})

        # 4. distance_from_center
        city_center = CITY_CENTERS[city]
        def get_distance(row):
            if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')):
                return geodesic((row['latitude'], row['longitude']), city_center).km
            return np.nan
        df['distance_from_center'] = df.apply(get_distance, axis=1)

        # 6 & 7. Mjeseci od datuma
        if 'host_since' in df.columns:
            df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce')
            df['host_age_years'] = (current_date - df['host_since']).dt.days / 365.25
        if 'first_review' in df.columns:
            df['first_review'] = pd.to_datetime(df['first_review'], errors='coerce')
            df['listing_age_years'] = (current_date - df['first_review']).dt.days / 365.25

        # 9. Dodaje city kolonu
        df['city'] = city

        # 10. price_usd
        df['price_usd'] = conversion_rates[city](df['price'])

        # 8. Uklanja ekstremne outliere u cijeni po room_type
        if 'room_type' in df.columns and not df.empty:
            valid_groups = []
            for r_type, group in df.groupby('room_type'):
                Q1 = group['price'].quantile(0.25)
                if pd.isna(Q1):
                    valid_groups.append(group)
                else:
                    Q3 = group['price'].quantile(0.75)
                    IQR = Q3 - Q1
                    upper_bound = Q3 + 1.5 * IQR
                    valid_group = group[(group['price'] <= upper_bound) | (group['price'].isna())]
                    valid_groups.append(valid_group)
            if valid_groups:
                df = pd.concat(valid_groups)
            else:
                df = df.iloc[0:0]
        elif df.empty:
            pass # Već je prazan, zadržaće kolone

        outlier_free_rows = len(df)
        print(f"    - Opservacije nakon sanacije ekstremnih odstupanja (outliers): {outlier_free_rows}")

        if not df.empty:
            avg_price = df['price'].mean()
            avg_usd = df['price_usd'].mean()
            avg_dist = df['distance_from_center'].mean()
            print(f"    - Prosječna cijena (lokalno): {avg_price:.0f} | (USD ekvivalent): {avg_usd:.0f}")
            print(f"    - Prosječna stvarna udaljenost od epicentra grada (km): {avg_dist:.1f}")
        else:
            print(f"    - Prethodna filtriranja su rezultirala sa nula opservacija.")

        city_dfs[city] = df
        all_dfs.append(df)

    combined_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    return city_dfs, combined_df
