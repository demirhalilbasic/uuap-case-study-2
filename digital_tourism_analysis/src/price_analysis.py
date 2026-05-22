"""
price_analysis.py
-----------------
Analiza distribucije cijena Airbnb listinga po gradu i tipu smještaja.
Generiše grafike 1.x i 4.x.

FIX: FacetGrid se sada sprema koristeći g.fig.savefig() umjesto plt.savefig(),
     što je bio uzrok neispravnog snimanja grafika 1.3.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from src.utils import CITY_COLORS, CITY_CURRENCIES, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


def run_price_analysis(city_dfs, all_df):
    out_dir_1 = 'output/01_price_distribution'
    out_dir_3 = 'output/03_room_type'
    ensure_dir_exists(out_dir_1)
    ensure_dir_exists(out_dir_3)

    # ------------------------------------------------------------------
    # GRAF 1.1 — Distribucija cijena po gradu (6 subplots, histogram+KDE)
    # ------------------------------------------------------------------
    print("    - Generisanje histograma distribucije cijena za sve gradove...")
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    for i, (city, df) in enumerate(city_dfs.items()):
        ax = axes[i]
        currency = CITY_CURRENCIES.get(city, '')
        price_data = df['price'].dropna()

        sns.histplot(price_data, kde=True, ax=ax, color=CITY_COLORS[city], bins=50)

        mean_val   = price_data.mean()
        median_val = price_data.median()
        std_val    = price_data.std()

        ax.axvline(mean_val,   color='red',   linestyle='--', linewidth=1.5,
                   label=f'Srednja: {mean_val:.0f}')
        ax.axvline(median_val, color='green', linestyle='-',  linewidth=1.5,
                   label=f'Medijan: {median_val:.0f}')

        ax.set_title(f"{city}  (n={len(price_data):,})", fontsize=12)
        ax.set_xlabel(f'Cijena ({currency})')
        ax.set_ylabel('Broj listinga')
        ax.legend(fontsize=9)

        ax.text(0.97, 0.95, f'Std: {std_val:.0f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9, color='grey')

    plt.suptitle('Distribucija cijena Airbnb smještaja po gradu', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_1, '1_1_distribution_by_city.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # GRAF 1.2 — Box plot cijena u USD (svi gradovi, sortirani po medianu)
    # ------------------------------------------------------------------
    print(f"    - Kreiranje box plot komparacije u USD za {len(all_df):,} listinga...")
    plt.figure(figsize=(14, 8))
    order = (all_df.groupby('city')['price_usd']
             .median().sort_values().index.tolist())

    sns.boxplot(
        data=all_df, x='city', y='price_usd', order=order,
        hue='city', palette=CITY_COLORS, legend=False,
        flierprops={"marker": "o", "alpha": 0.2, "markersize": 3}
    )
    plt.yscale('log')
    plt.title('Komparacija cijena između gradova (USD, logaritamska skala)', fontsize=14)
    plt.xlabel('Grad')
    plt.ylabel('Cijena po noćenju (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_1, '1_2_boxplot_usd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Graf 1.3 — Violin plot cijena po gradu i room_type
    # FIX: koristi g.fig.savefig() jer je FacetGrid odvojen objekt od plt
    # ------------------------------------------------------------------
    print("    - Generisanje violin plot distribucija unutar tipa smještaja...")
    plot_df = all_df[all_df['price_usd'] > 0].copy()

    g = sns.FacetGrid(
        plot_df, row='city', col='room_type',
        margin_titles=True, height=3, aspect=1.5, sharey=False
    )
    g.map_dataframe(sns.violinplot, y='price_usd', color='steelblue', inner='quartile')
    g.set_axis_labels('', 'Cijena (USD)')
    g.fig.suptitle('Distribucija cijena po tipu smještaja i gradu', y=1.02, fontsize=14)

    # ISPRAVNO: g.fig.savefig, NE plt.savefig
    g.fig.savefig(
        os.path.join(out_dir_3, '1_3_violin_room_type.png'),
        dpi=150, bbox_inches='tight'
    )
    plt.close('all')

    # ------------------------------------------------------------------
    # Graf 1.4 — Broj soba vs. cijena (Entire home/apt)
    # ------------------------------------------------------------------
    print("    - Analiziranje korelacije cijene u odnosu na broj soba...")
    entire_home = all_df[
        (all_df['room_type'] == 'Entire home/apt') &
        all_df['bedrooms'].notna() &
        (all_df['bedrooms'] > 0)
    ].copy()

    if not entire_home.empty:
        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=entire_home.sample(min(8000, len(entire_home)), random_state=42),
            x='bedrooms', y='price_usd',
            hue='city', palette=CITY_COLORS, alpha=0.5, s=30
        )
        plt.title('Cijena u odnosu na broj soba — cijeli stanovi', fontsize=14)
        plt.xlabel('Broj spavaćih soba')
        plt.ylabel('Cijena (USD)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_3, '1_4_price_per_bedroom.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 1.5 — Prosječna cijena po osobi po gradu
    # ------------------------------------------------------------------
    print("    - Proračun i vizualizacija prosječne cijene po osobi...")
    df_copy = all_df.copy()
    df_copy = df_copy[df_copy['accommodates'] > 0]
    df_copy['price_per_person'] = df_copy['price_usd'] / df_copy['accommodates']

    avg_pp = (df_copy.groupby('city')['price_per_person']
              .mean().sort_values(ascending=False).reset_index())

    plt.figure(figsize=(12, 7))
    sns.barplot(
        data=avg_pp, x='city', y='price_per_person',
        hue='city', palette=CITY_COLORS, legend=False
    )
    plt.title('Prosječna cijena po osobi po gradu (USD)', fontsize=14)
    plt.xlabel('Grad')
    plt.ylabel('Prosječna cijena po osobi (USD)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_3, '1_5_price_per_person.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def run_cross_city_comparison(all_df):
    out_dir_4 = 'output/04_city_comparison'
    ensure_dir_exists(out_dir_4)

    # ------------------------------------------------------------------
    # Graf 4.1 — Prosječna cijena USD po gradu i room_type (grouped bar)
    # ------------------------------------------------------------------
    print("    - Prebrojavanje parametara za međugradsku komparaciju cijena...")
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=all_df, x='city', y='price_usd',
        hue='room_type', errorbar=None
    )
    plt.title('Prosječna Airbnb cijena u USD — komparacija gradova i tipova smještaja',
              fontsize=14)
    plt.xlabel('Grad')
    plt.ylabel('Prosječna cijena (USD)')
    plt.legend(title='Tip smještaja', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_4, '4_1_avg_price_usd.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Graf 4.2 — Struktura ponude: 100% stacked bar po room_type i gradu
    # ------------------------------------------------------------------
    print("    - Formiranje 100% stacked bar strukture ponude...")
    crosstab = pd.crosstab(all_df['city'], all_df['room_type'], normalize='index') * 100
    ax = crosstab.plot(kind='bar', stacked=True, figsize=(14, 8), colormap='tab10')
    plt.title('Struktura ponude Airbnb smještaja po gradu (%)', fontsize=14)
    plt.ylabel('Udio (%)')
    plt.xlabel('Grad')
    plt.xticks(rotation=30, ha='right')
    plt.legend(title='Tip smještaja', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_4, '4_2_market_structure.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # ------------------------------------------------------------------
    # Graf 4.3 — Pairplot ključnih varijabli
    # ------------------------------------------------------------------
    print("    - Plotanje scatter matrix preglednika osnovnih faktora...")
    cols = ['price_usd', 'distance_from_center',
            'review_scores_rating', 'amenity_count', 'accommodates']
    valid_cols = [c for c in cols if c in all_df.columns]

    if len(valid_cols) >= 3:
        sample = all_df[valid_cols + ['city']].dropna().sample(
            min(2000, len(all_df)), random_state=42
        )
        g = sns.pairplot(
            sample, vars=valid_cols, hue='city',
            palette=CITY_COLORS, corner=True,
            plot_kws={'alpha': 0.4, 's': 15}
        )
        g.fig.suptitle('Pairplot ključnih varijabli — relacije između faktora',
                       y=1.02, fontsize=14)
        g.fig.savefig(os.path.join(out_dir_4, '4_3_scatter_matrix.png'),
                      dpi=150, bbox_inches='tight')
        plt.close('all')

    # ------------------------------------------------------------------
    # GRAF 4.4 — Bubble chart: cijena vs. percipirana vrijednost
    # ------------------------------------------------------------------
    print("    - Agregiranje komparacije cijene i vrijednosti za novac...")
    if 'review_scores_value' in all_df.columns:
        city_stats = all_df.groupby('city').agg(
            avg_price=('price_usd', 'mean'),
            avg_val=('review_scores_value', 'mean'),
            count=('id', 'count')
        ).reset_index()

        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=city_stats, x='avg_val', y='avg_price',
            size='count', sizes=(200, 2000),
            hue='city', palette=CITY_COLORS, legend=False
        )
        for _, row in city_stats.iterrows():
            plt.annotate(
                row['city'],
                (row['avg_val'], row['avg_price']),
                textcoords='offset points', xytext=(8, 4), fontsize=11
            )
        plt.title('Cijena vs. percipirana vrijednost — koji grad nudi najviše za novac?',
                  fontsize=14)
        plt.xlabel('Prosječna ocjena vrijednosti za novac')
        plt.ylabel('Prosječna cijena (USD)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir_4, '4_4_value_for_money.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()
