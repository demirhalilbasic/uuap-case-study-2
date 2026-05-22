"""
hypothesis_testing.py
---------------------
Testiranje glavne hipoteze:
  "Lokacija unutar grada (blizina centru) je primarniji prediktor cijene
   Airbnb smještaja od tipa smještaja — ali ne vrijedi jednako svugdje."

Metode:
  - OLS višestruka regresija s log(price) kao zavisnom varijablom
  - ANOVA (one-way) za uticaj room_type na cijenu
  - Kruskal-Wallis (neparametrijski) za razlike između kvartova
  - Spearman korelacijska matrica

FIX: Graf 7.2 sada prikazuje apsolutne vrijednosti (abs()) β koeficijenata,
     jer negativni koeficijent ne znači da faktor nema uticaj —
     samo znači suprotni smjer efekta.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from scipy import stats
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


def run_hypothesis_testing(city_dfs, all_df):
    out_dir      = 'output/07_hypothesis'
    out_dir_reg  = os.path.join(out_dir, 'regression')
    out_dir_anova= os.path.join(out_dir, 'anova_and_kruskal')
    out_dir_corr = os.path.join(out_dir, 'correlations')
    ensure_dir_exists(out_dir_reg)
    ensure_dir_exists(out_dir_anova)
    ensure_dir_exists(out_dir_corr)

    # ------------------------------------------------------------------
    # Analiza 7.1 — OLS višestruka regresija po gradu
    # ------------------------------------------------------------------
    print("    - Sprovođenje OLS regresije po gradovima...")
    required_cols = [
        'price_usd', 'distance_from_center', 'room_type',
        'accommodates', 'bedrooms', 'amenity_count',
        'host_is_superhost', 'review_scores_location', 'review_scores_rating'
    ]
    std_coefs_list = []

    for city, df in city_dfs.items():
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            print(f"    Upozorenje: {city}: nedostaju kolone {missing}. Preskakanje regresije.")
            continue

        analysis_df = df[required_cols].dropna().copy()
        if len(analysis_df) < 50:
            print(f"    Upozorenje: {city}: premalo opservacija ({len(analysis_df)}). Preskakanje.")
            continue

        analysis_df['log_price'] = np.log(analysis_df['price_usd'])
        analysis_df['host_is_superhost'] = analysis_df['host_is_superhost'].astype(float)

        # Standardizacija numeričkih prediktora (za usporedive β koeficijente)
        num_preds = [
            'distance_from_center', 'accommodates', 'bedrooms',
            'amenity_count', 'review_scores_location', 'review_scores_rating'
        ]
        for col in num_preds:
            std = analysis_df[col].std()
            if std > 0:
                analysis_df[col] = (analysis_df[col] - analysis_df[col].mean()) / std

        formula = (
            'log_price ~ distance_from_center + C(room_type) + accommodates '
            '+ bedrooms + amenity_count + host_is_superhost '
            '+ review_scores_location + review_scores_rating'
        )

        try:
            model = smf.ols(formula, data=analysis_df).fit()

            # Spremi OLS summary kao .txt
            with open(
                os.path.join(out_dir_reg, f'7_1_{city}_regression_summary.txt'),
                'w', encoding='utf-8'
            ) as f:
                f.write(model.summary().as_text())

            coefs = model.params
            coefs.name = city
            std_coefs_list.append(coefs)

            beta_dist = coefs.get('distance_from_center', 0)
            beta_room = coefs.get('C(room_type)[T.Private room]', 0)

            # Poređenje apsolutnih vrijednosti (smjer efekta nije bitan za dominaciju)
            dominance = "LOKACIJA DOMINIRA" if abs(beta_dist) > abs(beta_room) \
                        else "TIP SMJEŠTAJA DOMINIRA"
            print(f"    - {city}: β_distance={beta_dist:.3f}, "
                  f"β_room_private={beta_room:.3f} → {dominance}")

        except Exception as e:
            print(f"    Greška u regresiji za {city}: {e}")

    # ------------------------------------------------------------------
    # Graf 7.1 — Heatmap standardiziranih koeficijenata
    # ------------------------------------------------------------------
    if std_coefs_list:
        std_coefs_df = pd.concat(std_coefs_list, axis=1).T
        if 'Intercept' in std_coefs_df.columns:
            std_coefs_df = std_coefs_df.drop(columns=['Intercept'])

        # Čišćenje naziva kolona (statsmodels dodaje C(room_type)[T.x] prefix)
        std_coefs_df.columns = [
            c.replace('C(room_type)[T.', 'room: ').replace(']', '')
            for c in std_coefs_df.columns
        ]

        plt.figure(figsize=(18, 8))
        sns.heatmap(
            std_coefs_df.astype(float), cmap='coolwarm', center=0,
            annot=True, fmt='.2f', linewidths=0.5
        )
        plt.title('Standardizirani β koeficijenti regresije — koji faktori određuju cijenu?',
                  fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_reg, '7_1_coefficients_heatmap.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # ------------------------------------------------------------------
        # Graf 7.2 — Lokacija vs. Tip smještaja (apsolutni β koeficijenti)
        # FIX: koristi .abs() kako bi se negativni β prikazali kao uticaj,
        #      a ne kao suprotni smjer na grafikonu
        # ------------------------------------------------------------------
        print("    - Generisanje barchart pregleda lokacija vs. tip smještaja...")

        # Pronađi ispravno ime kolone za private room (može varirati)
        room_col_candidates = [
            c for c in std_coefs_df.columns if 'private' in c.lower()
        ]
        room_col = room_col_candidates[0] if room_col_candidates else None

        if room_col and 'distance_from_center' in std_coefs_df.columns:
            loc_vs_room = pd.DataFrame({
                'Uticaj lokacije |β_distance|':
                    std_coefs_df['distance_from_center'].astype(float).abs(),
                'Uticaj tipa smještaja |β_private_room|':
                    std_coefs_df[room_col].astype(float).abs()
            })

            ax = loc_vs_room.plot(kind='bar', figsize=(14, 8),
                                  color=['#2ca02c', '#1f77b4'])
            plt.title('Lokacija vs. tip smještaja: što više određuje cijenu Airbnb-a?',
                      fontsize=14)
            plt.ylabel('Apsolutna vrijednost standardiziranog β koeficijenta')
            plt.xlabel('Grad')
            plt.xticks(rotation=30, ha='right')
            plt.axhline(0, color='black', linewidth=0.8)
            plt.legend(loc='upper right')

            # Anotacija: koji faktor dominira za svaki grad
            for i, city in enumerate(loc_vs_room.index):
                loc_val  = loc_vs_room.loc[city, 'Uticaj lokacije |β_distance|']
                room_val = loc_vs_room.loc[city, 'Uticaj tipa smještaja |β_private_room|']
                label = 'LOK.' if loc_val > room_val else 'TIP'
                plt.text(i, max(loc_val, room_val) + 0.01, label,
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_anova, '7_2_location_vs_room_type.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Analiza 7.2 — ANOVA: room_type vs. cijena
    # ------------------------------------------------------------------
    print("    - Izvršavanje ANOVA procjene uticaja tipa smještaja...")
    with open(os.path.join(out_dir_anova, '7_2_ANOVA_results.txt'),
              'w', encoding='utf-8') as f:
        f.write("ANOVA test: uticaj tipa smještaja na cijenu\n")
        f.write("=" * 50 + "\n\n")
        for city, df in city_dfs.items():
            if 'room_type' not in df.columns:
                continue
            groups = [
                g['price_usd'].dropna()
                for _, g in df.groupby('room_type')
                if len(g['price_usd'].dropna()) > 4
            ]
            if len(groups) > 1:
                try:
                    f_stat, p_val = stats.f_oneway(*groups)
                    line = f"{city}: F={f_stat:.2f}, p={p_val:.3e}\n"
                    print(f"    - ANOVA ({city}): F={f_stat:.2f}, p={p_val:.3e}")
                    f.write(line)
                except Exception as e:
                    f.write(f"{city}: greška — {e}\n")

    # ------------------------------------------------------------------
    # Analiza 7.3 — Spearman korelacijska matrica po gradu
    # ------------------------------------------------------------------
    print("    - Kreiranje Spearman korelacijskih matrica...")
    corr_cols = [
        'distance_from_center', 'price_usd', 'accommodates', 'bedrooms',
        'amenity_count', 'review_scores_location', 'review_scores_rating',
        'review_scores_value', 'reviews_per_month', 'availability_365'
    ]

    for city, df in city_dfs.items():
        avail = [c for c in corr_cols if c in df.columns]
        if len(avail) < 3:
            continue
        corr_matrix = (df[avail]
                       .apply(pd.to_numeric, errors='coerce')
                       .corr(method='spearman'))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm',
                    fmt='.2f', vmin=-1, vmax=1, linewidths=0.3)
        plt.title(f'Spearman korelacijska matrica varijabli — {city}', fontsize=13)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_corr, f'7_3_correlation_{city}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Analiza 7.4 — Kruskal-Wallis: razlike između kvartova
    # ------------------------------------------------------------------
    print("    - Izvođenje Kruskal-Wallis neparametrijskog testa...")
    kw_results = []
    for city, df in city_dfs.items():
        if 'neighbourhood_cleansed' not in df.columns:
            continue
        groups = [
            g['price_usd'].dropna()
            for _, g in df.groupby('neighbourhood_cleansed')
            if len(g['price_usd'].dropna()) > 4
        ]
        if len(groups) > 1:
            try:
                h_stat, p_val = stats.kruskal(*groups)
                kw_results.append({'Grad': city, 'H': h_stat, 'p': p_val})
                print(f"    - Kruskal-Wallis ({city}): H={h_stat:.2f}, p={p_val:.3e}")
            except Exception as e:
                print(f"    Upozorenje: Kruskal-Wallis greška za {city}: {e}")

    if kw_results:
        kw_df = pd.DataFrame(kw_results)
        kw_df.to_csv(
            os.path.join(out_dir_anova, '7_4_kruskal_wallis_results.csv'),
            index=False, encoding='utf-8'
        )
