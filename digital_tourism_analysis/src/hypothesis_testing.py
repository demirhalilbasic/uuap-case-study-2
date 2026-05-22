import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_hypothesis_testing(city_dfs, all_df):
    """
    Sprovodi OLS višestruku regresiju, ANOVA i Kruskal-Wallis testove
    kako bi ispitali glavnu hipotezu o uticaju lokacije vs. tipa smještaja na cijenu.
    """
    out_dir = 'output/07_hypothesis'

    out_dir_reg = os.path.join(out_dir, 'regression')
    out_dir_anova = os.path.join(out_dir, 'anova_and_kruskal')
    out_dir_corr = os.path.join(out_dir, 'correlations')

    ensure_dir_exists(out_dir_reg)
    ensure_dir_exists(out_dir_anova)
    ensure_dir_exists(out_dir_corr)

    print("    - Sprovođenje rigoroznih statističkih testova i OLS regresije po gradovima...")

    std_coefs_list = []

    # Prilagodba i priprema podataka za OLS model po svakom gradu
    for city, df in city_dfs.items():
        req_cols = ['price_usd', 'distance_from_center', 'room_type', 'accommodates',
                    'bedrooms', 'amenity_count', 'host_is_superhost',
                    'review_scores_location', 'review_scores_rating']

        missing = [c for c in req_cols if c not in df.columns]
        if missing:
            continue

        analysis_df = df[req_cols].dropna().copy()
        if len(analysis_df) < 50:
            continue

        analysis_df['log_price'] = np.log(analysis_df['price_usd'])
        # Konvertovanje boolean vrijednosti u integere za potrebe regresije
        analysis_df['host_is_superhost'] = analysis_df['host_is_superhost'].astype(int)

        # Standardizacija neprekidnih prediktora (za komparaciju uticaja beta koeficijenata)
        num_preds = ['distance_from_center', 'accommodates', 'bedrooms',
                     'amenity_count', 'review_scores_location', 'review_scores_rating']

        for col in num_preds:
            analysis_df[col] = (analysis_df[col] - analysis_df[col].mean()) / analysis_df[col].std()

        formula = 'log_price ~ distance_from_center + C(room_type) + accommodates + bedrooms + amenity_count + host_is_superhost + review_scores_location + review_scores_rating'

        try:
            model = smf.ols(formula, data=analysis_df).fit()

            # Spašavamo OLS summary u txt formatu (Analiza 7.1)
            with open(os.path.join(out_dir_reg, f'7_1_{city}_regression_summary.txt'), 'w', encoding='utf-8') as f:
                f.write(model.summary().as_text())

            coefs = model.params
            coefs.name = city
            std_coefs_list.append(coefs)

            # Izdvajanje ključnih koeficijenata za zapisivanje nalaza hipoteze
            beta_dist = coefs.get('distance_from_center', 0)
            beta_room = coefs.get('C(room_type)[T.Private room]', 0)

            dominance_str = "LOKACIJA DOMINIRA" if abs(beta_dist) > abs(beta_room) else "TIP SMJEŠTAJA DOMINIRA"
            print(f"    - {city}: β_distance={beta_dist:.3f}, β_room_private={beta_room:.3f} -> {dominance_str}")

        except Exception as e:
            print(f"    Greška prilikom kalkulacije regresije za {city}: {e}")

    # Aggregacija koeficijenata za Heatmap (Graf 7.1)
    if std_coefs_list:
        std_coefs_df = pd.concat(std_coefs_list, axis=1).T
        if 'Intercept' in std_coefs_df.columns:
            std_coefs_df = std_coefs_df.drop(columns=['Intercept'])

        plt.figure(figsize=(16, 8))
        sns.heatmap(std_coefs_df, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=.5)
        plt.title('Standardizirani koeficijenti regresije — koji faktori određuju cijenu?')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_reg, '7_1_coefficients_heatmap.png'), dpi=150)
        plt.close()

        print("    - Generisanje barchart pregleda lokacija vs. tip smještaja...")
        # Graf 7.2 — "Lokacija vs. Tip smještaja"
        # OLS models code C(room_type)[T.Private room] internally via statsmodels smf module
        room_col = 'C(room_type)[T.Private room]' if 'C(room_type)[T.Private room]' in std_coefs_df.columns else 'room_type_Private room'

        loc_vs_room = std_coefs_df[['distance_from_center', room_col]].copy()
        loc_vs_room.columns = ['Uticaj Lokacije (|β_distance|)', 'Uticaj Tipa Smještaja (|β_private_room|)']

        ax = loc_vs_room.plot(kind='bar', figsize=(14, 8), color=['#2ca02c', '#1f77b4'])
        plt.title('Lokacija vs. tip smještaja: što više određuje cijenu?')
        plt.ylabel('Apsolutna vrijednost standardiziranog koeficijenta')
        plt.axhline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_anova, '7_2_location_vs_room_type.png'), dpi=150)
        plt.close()

    print("    - Izvršavanje ANOVA procjene uticaja tipa smještaja...")
    # Analiza 7.2 — ANOVA test: room_type vs. cijena
    with open(os.path.join(out_dir_anova, '7_2_ANOVA_results.txt'), 'w', encoding='utf-8') as f:
        f.write("ANOVA test: uticaj tipa smještaja na cijenu\n")
        f.write("="*50 + "\n")
        for city, df in city_dfs.items():
            if 'room_type' in df.columns:
                groups = [group['price_usd'].dropna() for name, group in df.groupby('room_type')]
                if len(groups) > 1:
                    try:
                        f_stat, p_val = stats.f_oneway(*groups)
                        print(f"    - ANOVA ({city}): F-stat={f_stat:.2f}, p-val={p_val:.3e}")
                        f.write(f"{city}: ANOVA F={f_stat:.2f}, p={p_val:.3e}\n")
                    except Exception:
                        pass

    print("    - Kreiranje Spearman korelacijskih matrica...")
    # Analiza 7.3 — Korelacijska matrica (po gradu)
    corr_cols = ['distance_from_center', 'price_usd', 'accommodates', 'bedrooms', 'amenity_count',
                 'review_scores_location', 'review_scores_rating', 'review_scores_value',
                 'reviews_per_month', 'availability_365']
    for city, df in city_dfs.items():
        avail_vars = [v for v in corr_cols if v in df.columns]
        if len(avail_vars) > 1:
            corr_matrix = df[avail_vars].apply(pd.to_numeric, errors='coerce').corr(method='spearman')
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
            plt.title(f'Spearman korelacijska matrica varijabli — {city}')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir_corr, f'7_3_correlation_{city}.png'), dpi=150)
            plt.close()

    print("    - Izvođenje Kruskal-Wallis neparametrijskog testa...")
    for city, df in city_dfs.items():
        if 'neighbourhood_cleansed' in df.columns:
            groups = [group['price_usd'].dropna() for name, group in df.groupby('neighbourhood_cleansed') if len(group['price_usd'].dropna()) > 4]
            if len(groups) > 1:
                try:
                    h_stat, p_val = stats.kruskal(*groups)
                    print(f"    - Kruskal-Wallis ({city}): H-stat={h_stat:.2f}, p-val={p_val:.3e}")
                except Exception as e:
                    print(f"    - Kruskal-Wallis greška za {city}: {e}")
