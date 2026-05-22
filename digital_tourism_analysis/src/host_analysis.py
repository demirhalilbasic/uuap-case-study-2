import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_host_analysis(city_dfs, all_df):
    """
    Analiza elemenata povezanih sa iznajmljivačima (hostovima)
    i utvrđivanje uticaja njihovog statusa i iskustva na performanse.
    """
    out_dir = 'output/05_host_analysis'
    ensure_dir_exists(out_dir)

    print("    - Validacija Superhost statusa i njihovog premium iznosa naplate...")
    # Graf 5.1 — Superhost premium
    if 'host_is_superhost' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=all_df, x='city', y='price_usd', hue='host_is_superhost', palette='Set2')
        plt.yscale('log')
        plt.title('Superhost premium — koliko više naplaćuju certificirani hostovi?')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_1_superhost_premium.png'), dpi=150)
        plt.close()

    print("    - Klasificiranje hostova (privatni vs. komercijalni)...")
    # Graf 5.2 — Distribucija broja listinga po hostu
    if 'calculated_host_listings_count' in all_df.columns:
        plt.figure(figsize=(14, 8))
        
        def categorize_host(cnt):
            if pd.isna(cnt): return 'Nepoznato'
            if cnt == 1: return 'Privatni (1)'
            if cnt <= 5: return 'Mali (2-5)'
            if cnt <= 20: return 'Veliki (6-20)'
            return 'Komercijalni (20+)'
            
        all_df['host_type'] = all_df['calculated_host_listings_count'].apply(categorize_host)
        
        sns.histplot(data=all_df, x='calculated_host_listings_count', hue='city', element='poly', 
                     log_scale=True, palette=CITY_COLORS, fill=False)
        plt.title('Privatni vs. komercijalni hostovi — distribucija po gradu')
        plt.xlabel('Broj listinga (log skala)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_2_host_listings_distribution.png'), dpi=150)
        plt.close()

    print("    - Evaluacija zavisnosti 'Instant Bookable' dozvola...")
    # Graf 5.4 — Instant Bookable vs. cijena i ocjene
    if 'instant_bookable' in all_df.columns:
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        sns.barplot(data=all_df, x='instant_bookable', y='price_usd', ax=axes[0], errorbar=('ci', 95))
        axes[0].set_title('Uticaj na prosječnu cijenu (USD)')
        
        sns.barplot(data=all_df, x='instant_bookable', y='review_scores_rating', ax=axes[1], errorbar=('ci', 95))
        axes[1].set_title('Uticaj na prosječne ostvarene ocjene')
        axes[1].set_ylim(4.0, 5.0) 
        
        if 'availability_365' in all_df.columns:
            sns.barplot(data=all_df, x='instant_bookable', y='availability_365', ax=axes[2], errorbar=('ci', 95))
            axes[2].set_title('Uticaj na stepen raspoloživosti jedinice')
            
        plt.suptitle('Instant rezervacija mogućnost — razlika u cijeni, ocjenama i raspoloživosti', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_4_instant_bookable.png'), dpi=150)
        plt.close()

    print("    - Mjerenje uticaja historijskog iskustva (godine na platformi)...")
    # Graf 5.5 — Starost hosta vs. prosječna ocjena
    if 'host_age_years' in all_df.columns and 'review_scores_rating' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=all_df.sample(min(10000, len(all_df))), x='host_age_years', y='review_scores_rating', 
                        hue='city', alpha=0.4, palette=CITY_COLORS)
        plt.title('Iskustvo hosta (godine od registracije) i kvalitet pružene usluge')
        plt.xlabel('Godine iskustva na platformi')
        plt.ylabel('Ukupna ocjena')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '5_5_host_age_vs_rating.png'), dpi=150)
        plt.close()
