"""
host_analysis.py
----------------
Analiza karakteristika iznajmljivača (hostova):
  - Superhost premium
  - Distribucija privatnih vs. komercijalnih hostova
  - Uticaj veličine portfolia na cijenu (DODAN GRAF 5.3 koji je nedostajao)
  - Instant Bookable vs. cijena/ocjene/dostupnost
  - Iskustvo hosta (godine) vs. ocjena

FIX: Dodan Graf 5.3 koji je bio definisan u specifikaciji ali nedostajao
     u originalnom kodu.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


def run_host_analysis(city_dfs, all_df):
    out_dir = 'output/05_host_analysis'
    ensure_dir_exists(out_dir)

    # ------------------------------------------------------------------
    # Graf 5.1 — Superhost premium: cijena Superhost vs. Regular host
    # ------------------------------------------------------------------
    print("    - Validacija Superhost statusa i njihovog cjenovnog premijuma...")
    if 'host_is_superhost' in all_df.columns:
        df_super = all_df[all_df['host_is_superhost'].notna()].copy()
        df_super['Status hosta'] = df_super['host_is_superhost'].map(
            {True: 'Superhost', False: 'Regularni host'}
        )

        plt.figure(figsize=(14, 8))
        sns.boxplot(
            data=df_super, x='city', y='price_usd',
            hue='Status hosta', palette=['#FFD700', '#A0A0A0']
        )
        plt.yscale('log')
        plt.title('Superhost premium — koliko više naplaćuju certificirani hostovi?',
                  fontsize=14)
        plt.xlabel('Grad')
        plt.ylabel('Cijena po noćenju (USD, log skala)')
        plt.legend(title='Status hosta')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_1_superhost_premium.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 5.2 — Distribucija broja listinga po hostu (privatni vs. komercijalni)
    # ------------------------------------------------------------------
    print("    - Klasificiranje hostova (privatni vs. komercijalni)...")
    if 'calculated_host_listings_count' in all_df.columns:
        def categorize_host(cnt):
            if pd.isna(cnt):     return 'Nepoznato'
            if cnt == 1:          return '1 — Privatni'
            if cnt <= 5:          return '2–5 — Mali'
            if cnt <= 20:         return '6–20 — Veliki'
            return '20+ — Komercijalni'

        all_df['host_type'] = all_df['calculated_host_listings_count'].apply(categorize_host)

        plt.figure(figsize=(14, 8))
        sns.histplot(
            data=all_df,
            x='calculated_host_listings_count',
            hue='city', element='poly',
            log_scale=True, palette=CITY_COLORS, fill=False, linewidth=2
        )
        plt.title('Privatni vs. komercijalni hostovi — distribucija po gradu',
                  fontsize=14)
        plt.xlabel('Broj listinga hosta (log skala)')
        plt.ylabel('Broj hostova')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_2_host_listings_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 5.3 — Komercijalni hostovi vs. cijena (NEDOSTAJAO U ORIGINALU)
    # Prikazuje scatter: broj listinga hosta vs. cijena, po gradu
    # Pitanje: naplaćuju li komercijalni hostovi više ili manje?
    # ------------------------------------------------------------------
    print("    - Evaluacija zavisnosti veličine portfolia hosta i cijene (Graf 5.3)...")
    if 'calculated_host_listings_count' in all_df.columns:
        sample = all_df[
            all_df['calculated_host_listings_count'].notna() &
            all_df['price_usd'].notna()
        ].sample(min(12000, len(all_df)), random_state=42)

        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=sample,
            x='calculated_host_listings_count', y='price_usd',
            hue='city', palette=CITY_COLORS, alpha=0.35, s=25
        )
        plt.xscale('log')
        plt.yscale('log')
        plt.title(
            'Portfolio veličina hosta vs. Cijena smještaja\n'
            'Naplaćuju li komercijalni hostovi više od privatnih?',
            fontsize=14
        )
        plt.xlabel('Broj listinga hosta (log skala)')
        plt.ylabel('Cijena po noćenju (USD, log skala)')
        plt.legend(title='Grad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_3_commercial_host_vs_price.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Dopunski barplot: prosječna cijena po tipu hosta
        if 'host_type' in all_df.columns:
            order = ['1 — Privatni', '2–5 — Mali', '6–20 — Veliki', '20+ — Komercijalni']
            order_exist = [o for o in order if o in all_df['host_type'].unique()]

            plt.figure(figsize=(12, 6))
            sns.barplot(
                data=all_df, x='host_type', y='price_usd',
                order=order_exist, errorbar=('ci', 95),
                color='steelblue'
            )
            plt.title('Prosječna cijena po kategoriji hosta (95% CI)', fontsize=13)
            plt.xlabel('Kategorija hosta')
            plt.ylabel('Prosječna cijena (USD)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, '5_3b_avg_price_by_host_type.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Graf 5.4 — Instant Bookable vs. cijena, ocjene, dostupnost
    # ------------------------------------------------------------------
    print("    - Evaluacija zavisnosti 'Instant Bookable' statusa...")
    if 'instant_bookable' in all_df.columns:
        df_ib = all_df[all_df['instant_bookable'].notna()].copy()
        df_ib['Instant rezervacija'] = df_ib['instant_bookable'].map(
            {True: 'Da', False: 'Ne'}
        )

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        # Use hue equal to x to allow passing a palette (avoids FutureWarning in
        # newer seaborn versions). Set dodge=False so bars are not split, then
        # remove the redundant legend created by the hue mapping.
        sns.barplot(data=df_ib, x='Instant rezervacija', y='price_usd',
                    ax=axes[0], errorbar=('ci', 95), hue='Instant rezervacija',
                    palette=['#2196F3', '#9E9E9E'], dodge=False)
        # remove auto legend created by hue to keep the plot clean
        if axes[0].get_legend() is not None:
            axes[0].get_legend().remove()
        axes[0].set_title('Prosječna cijena (USD)')
        axes[0].set_ylabel('USD')

        if 'review_scores_rating' in df_ib.columns:
            sns.barplot(data=df_ib, x='Instant rezervacija', y='review_scores_rating',
                        ax=axes[1], errorbar=('ci', 95), hue='Instant rezervacija',
                        palette=['#2196F3', '#9E9E9E'], dodge=False)
            if axes[1].get_legend() is not None:
                axes[1].get_legend().remove()
            axes[1].set_title('Prosječna ocjena gostiju')
            axes[1].set_ylim(4.0, 5.0)
            axes[1].set_ylabel('Ocjena (0–5)')

        if 'availability_365' in df_ib.columns:
            sns.barplot(data=df_ib, x='Instant rezervacija', y='availability_365',
                        ax=axes[2], errorbar=('ci', 95), hue='Instant rezervacija',
                        palette=['#2196F3', '#9E9E9E'], dodge=False)
            if axes[2].get_legend() is not None:
                axes[2].get_legend().remove()
            axes[2].set_title('Prosječna godišnja dostupnost (dani)')
            axes[2].set_ylabel('Dani')

        plt.suptitle(
            'Instant rezervacija — uticaj na cijenu, ocjene i raspoloživost',
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_4_instant_bookable.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 5.5 — Starost hosta (godine iskustva) vs. prosječna ocjena
    # ------------------------------------------------------------------
    print("    - Mjerenje uticaja iskustva hosta na kvalitet usluge...")
    if 'host_age_years' in all_df.columns and 'review_scores_rating' in all_df.columns:
        sample = all_df[
            all_df['host_age_years'].notna() &
            all_df['review_scores_rating'].notna()
        ].sample(min(10000, len(all_df)), random_state=42)

        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=sample,
            x='host_age_years', y='review_scores_rating',
            hue='city', alpha=0.35, palette=CITY_COLORS, s=25
        )
        # Regresijska linija kroz sve gradove
        sns.regplot(
            data=sample,
            x='host_age_years', y='review_scores_rating',
            scatter=False, color='black', line_kws={'linewidth': 2}
        )
        plt.title('Iskustvo hosta (godine na platformi) vs. Kvalitet pružene usluge',
                  fontsize=14)
        plt.xlabel('Godine iskustva na platformi')
        plt.ylabel('Ukupna ocjena gostiju')
        plt.ylim(3.5, 5.05)
        plt.legend(title='Grad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_5_host_age_vs_rating.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
