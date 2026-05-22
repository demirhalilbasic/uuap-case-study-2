import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from math import pi
from scipy.stats import spearmanr
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_review_analysis(city_dfs, all_df):
    """
    Analizira performativne ocjene smještaja i percepciju potrošača.
    Generiše profil kvaliteta po gradovima pomoću radar grafika.
    """
    out_dir = 'output/06_reviews_and_ratings'
    ensure_dir_exists(out_dir)

    print("    - Formiranje Radar Spiderchart vizualizacije po dimenzijama kvaliteta...")

    # Dimenzije ocjenjivanja
    review_cols = [
        'review_scores_accuracy', 'review_scores_cleanliness',
        'review_scores_checkin', 'review_scores_communication',
        'review_scores_location', 'review_scores_value'
    ]

    # Graf 6.1 — Radar chart (Spider) za prosječne ocjene po gradu
    has_radar_data = all(col in all_df.columns for col in review_cols)
    if has_radar_data:
        radar_df = all_df.groupby('city')[review_cols].mean().reset_index()

        categories = ['Tačnost procjene', 'Čistoća', 'Check-in proces', 'Komunikacija', 'Lokacija', 'Vrijednost za novac']
        N = len(categories)

        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        plt.xticks(angles[:-1], categories, size=12)

        ax.set_rlabel_position(0)
        plt.yticks([4.5, 4.6, 4.7, 4.8, 4.9], ["4.5", "4.6", "4.7", "4.8", "4.9"], color="grey", size=10)
        plt.ylim(4.5, 5.0)

        for _, row in radar_df.iterrows():
            city = row['city']
            values = row[review_cols].values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=city, color=CITY_COLORS[city])
            ax.fill(angles, values, alpha=0.1, color=CITY_COLORS[city])

        plt.title('Profil kvaliteta Airbnb tržišta po gradu (Prosječne ocjene)', size=16, y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '6_1_radar_chart_ratings.png'), dpi=150)
        plt.close()

    print("    - Regresijska analiza odnosa ocjene lokacije i stvarne cijene...")
    # Graf 6.2 — Review scores location vs. cijena
    if 'review_scores_location' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.regplot(data=all_df.sample(min(15000, len(all_df))), x='review_scores_location', y='price_usd',
                    scatter_kws={'alpha':0.2}, line_kws={'color':'red'}, lowess=True)
        plt.title('Ocjena lokacije od strane gostiju vs. Cijena - Plaćaju li gosti premijum za visokorangiranu lokaciju?')
        plt.xlabel('Ocjena lokacije')
        plt.ylabel('Cijena po noćenju (USD)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '6_2_location_vs_price.png'), dpi=150)
        plt.close()

    print("    - Analiza percipirane vrijednosti (Value rating) na odnosu cijene...")
    # Graf 6.3 — Review scores value vs. cijena
    if 'review_scores_value' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=all_df.sample(min(15000, len(all_df))), x='review_scores_value', y='price_usd',
                        hue='city', alpha=0.4, palette=CITY_COLORS)
        plt.title('Vrijednost za novac: Percipiraju li gosti skuplje smještaje kao kvalitetnije?')
        plt.xlabel('Ocjena vrijednosti (Value)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '6_3_value_vs_price.png'), dpi=150)
        plt.close()

    print("    - Priprema Spearman korelacijske toplotne mape za ugniježđene ocjenske cjeline...")
    # Graf 6.4 — Heatmap korelacije svih review scores dimenzija
    all_review_cols = ['review_scores_rating'] + review_cols
    if all(c in all_df.columns for c in all_review_cols):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        for i, (city, df) in enumerate(city_dfs.items()):
            valid_df = df[all_review_cols].dropna()
            if not valid_df.empty:
                corr = valid_df.corr(method='spearman')
                sns.heatmap(corr, ax=axes[i], annot=True, cmap='viridis', vmin=0, vmax=1, fmt=".2f", cbar=False)
                axes[i].set_title(city)
        plt.suptitle('Korelacija dimenzija ocjenjivanja unutar ocjenskog sistema — po gradu', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '6_4_reviews_correlation.png'), dpi=150)
        plt.close()

    print("    - Plotanje violin distribucije konačnih ocjena po geografskim sredinama...")
    # Graf 6.5 — Distribucija ukupnih ocjena (violin plot)
    if 'review_scores_rating' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.violinplot(data=all_df, x='city', y='review_scores_rating', hue='city', palette=CITY_COLORS, legend=False, inner='quartile')
        plt.title('Distribucija ukupnih ocjena gostiju — po gradu', fontsize=16)
        plt.savefig(os.path.join(out_dir, '6_5_ratings_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()
