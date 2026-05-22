import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
from src.utils import CITY_COLORS, CITY_CURRENCIES, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_price_analysis(city_dfs, all_df):
    out_dir_1 = 'output/01_price_distribution'
    ensure_dir_exists(out_dir_1)
    
    out_dir_3 = 'output/03_room_type'
    ensure_dir_exists(out_dir_3)

    out_dir_4 = 'output/04_city_comparison'
    ensure_dir_exists(out_dir_4)
    
    print("    - Generisanje histograma distribucije cijena za sve gradove...")
    # Graf 1.1 — Distribucija cijena po gradu (6 subplots)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for i, city in enumerate(city_dfs.keys()):
        df = city_dfs[city]
        sns.histplot(data=df, x='price', kde=True, ax=axes[i], color=CITY_COLORS[city], bins=50)
        mean_val = df['price'].mean()
        median_val = df['price'].median()
        axes[i].axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.2f}')
        axes[i].axvline(median_val, color='green', linestyle='-', label=f'Median: {median_val:.2f}')
        # Dodavanje oznake valute
        currency_label = CITY_CURRENCIES.get(city, '')
        axes[i].set_title(f"{city} (Valuta: {currency_label})")
        axes[i].legend()
        axes[i].set_xlabel(f'Cijena ({currency_label})')
    plt.suptitle('Distribucija cijena Airbnb smještaja po gradu', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_1, '1_1_distribution_by_city.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"    - Kreiranje box plot komparacije u USD za {len(all_df)} listinga...")
    # Graf 1.2 — Box plot cijena u USD (svi gradovi zajedno)
    plt.figure(figsize=(14, 8))
    order = all_df.groupby('city')['price_usd'].median().sort_values().index
    sns.boxplot(data=all_df, x='city', y='price_usd', order=order, hue='city', palette=CITY_COLORS, legend=False, flierprops={"marker": "o", "alpha": 0.3})
    plt.yscale('log')
    plt.title('Komparacija cijena između gradova (USD)')
    plt.savefig(os.path.join(out_dir_1, '1_2_boxplot_usd.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Generisanje violin plot distribucija unutar tipa smještaja...")
    # Graf 1.3 — Violin plot cijena po gradu i room_type
    g = sns.FacetGrid(all_df, row="city", col="room_type", margin_titles=True, height=3, aspect=1.5, sharey=False)
    g.map_dataframe(sns.violinplot, y="price_usd")
    g.fig.suptitle('Distribucija cijena po tipu smještaja i gradu', y=1.02)
    plt.savefig(os.path.join(out_dir_3, '1_3_violin_room_type.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Analiziranje korelacije cijene u odnosu na broj soba...")
    # Graf 1.4 — Cijena po sobi (price/bedrooms) za Entire home
    plt.figure(figsize=(14, 8))
    entire_home = all_df[all_df['room_type'] == 'Entire home/apt'].copy()
    entire_home['price_per_bedroom'] = entire_home['price_usd'] / entire_home['bedrooms']
    sns.scatterplot(data=entire_home, x='bedrooms', y='price_usd', hue='city', palette=CITY_COLORS, alpha=0.6)
    plt.title('Cijena u odnosu na broj soba — cijeli stanovi')
    plt.savefig(os.path.join(out_dir_3, '1_4_price_per_bedroom.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Proračun i vizualizacija prosječne cijene po osobi...")
    # Graf 1.5 — Cijena po osobi (price/accommodates)
    plt.figure(figsize=(14, 8))
    all_df_copy = all_df.copy()
    all_df_copy['price_per_person'] = all_df_copy['price_usd'] / all_df_copy['accommodates']
    avg_price_pp = all_df_copy.groupby('city')['price_per_person'].mean().sort_values(ascending=False).reset_index()
    sns.barplot(data=avg_price_pp, x='city', y='price_per_person', hue='city', palette=CITY_COLORS, legend=False)
    plt.title("Prosječna cijena po osobi po gradu (USD)", fontsize=16)
    plt.savefig(os.path.join(out_dir_3, '1_5_price_per_person.png'), dpi=150, bbox_inches='tight')
    plt.close()

def run_cross_city_comparison(all_df):
    out_dir_4 = 'output/04_city_comparison'
    ensure_dir_exists(out_dir_4)

    print("    - Prebrojavanje parametara za međugradsku komparaciju cijena...")
    # Graf 4.1 — Prosječna cijena (USD) po gradu i room_type
    plt.figure(figsize=(14, 8))
    sns.barplot(data=all_df, x='city', y='price_usd', hue='room_type', errorbar=None)
    plt.title('Prosječna Airbnb cijena u USD — komparacija gradova i tipova smještaja')
    plt.savefig(os.path.join(out_dir_4, '4_1_avg_price_usd.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Formiranje baze za 100% stacked bar strukture potražnje...")
    # Graf 4.2 — Broj listinga po room_type i gradu (100% stacked bar)
    plt.figure(figsize=(14, 8))
    crosstab = pd.crosstab(all_df['city'], all_df['room_type'], normalize='index') * 100
    crosstab.plot(kind='bar', stacked=True, figsize=(14, 8))
    plt.title('Struktura ponude Airbnb smještaja po gradu')
    plt.ylabel('Postotak (%)')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_4, '4_2_market_structure.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Plotanje scatter matrix preglednika osnovnih faktora...")
    # Graf 4.3 — Scatter matrix
    cols = ['price_usd', 'distance_from_center', 'review_scores_rating', 'amenity_count', 'accommodates']
    valid_cols = [c for c in cols if c in all_df.columns]
    if valid_cols:
        sns.pairplot(all_df.sample(min(2000, len(all_df))), vars=valid_cols, hue='city', palette=CITY_COLORS, corner=True)
        plt.suptitle('Pairplot ključnih varijabli — relacije između faktora', y=1.02)
        plt.savefig(os.path.join(out_dir_4, '4_3_scatter_matrix.png'), dpi=150, bbox_inches='tight')
        plt.close()

    print("    - Agregiranje parametara komparacije cijena i vrijednosti za novac...")
    # Graf 4.4 — Rangiranje gradova
    plt.figure(figsize=(14, 8))
    city_stats = all_df.groupby('city').agg(
        avg_price=('price_usd', 'mean'),
        avg_val=('review_scores_value', 'mean'),
        count=('id', 'count')
    ).reset_index()
    sns.scatterplot(data=city_stats, x='avg_val', y='avg_price', size='count', sizes=(100, 2000), hue='city', palette=CITY_COLORS, legend=False)
    for i in range(len(city_stats)):
        plt.text(city_stats['avg_val'].iloc[i], city_stats['avg_price'].iloc[i], city_stats['city'].iloc[i])
    plt.title('Cijena vs. percipirana vrijednost — koji grad nudi najviše za novac?')
    plt.savefig(os.path.join(out_dir_4, '4_4_value_for_money.png'), dpi=150, bbox_inches='tight')
    plt.close()
