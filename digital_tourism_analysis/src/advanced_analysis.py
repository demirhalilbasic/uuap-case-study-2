import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_advanced_analysis(city_dfs, all_df):
    """
    Sprovođenje naprednih istraživačkih metoda klasterizacije (KMeans),
    valorizacije prihoda, praćenja pogodnosti (amenities) i distribucije dozvola.
    """
    out_dir = 'output/08_advanced_analysis'
    ensure_dir_exists(out_dir)

    print("    - Konfigurisanje algoritmike i obavljanje K-Means klasterizacije tržišta...")

    # Graf 8.1 — KMeans Clustering listinga
    cluster_features = ['price_usd', 'distance_from_center', 'review_scores_rating', 'amenity_count', 'accommodates']
    if all(f in all_df.columns for f in cluster_features):
        cluster_df = all_df[cluster_features + ['city']].dropna().copy()
        if not cluster_df.empty:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_df[cluster_features])

            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            cluster_df['cluster'] = kmeans.fit_predict(scaled_data)

            plt.figure(figsize=(14, 8))
            sns.scatterplot(data=cluster_df.sample(min(15000, len(cluster_df))),
                            x='distance_from_center', y='price_usd', hue='cluster', palette='Set1', alpha=0.5)
            plt.yscale('log')
            plt.title('KMeans klasterizacija tržišnih segmenata Airbnb platforme (K=4)')
            plt.xlabel('Udaljenost od centra (km)')
            plt.ylabel('Cijena (USD)')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, '8_1_kmeans_clustering.png'), dpi=150)
            plt.close()

    print("    - Evaluacija komercijalnog dodataka popularnih pogodnosti (Amenities Premium)...")
    # Graf 8.2 — Uticaj amenities (pogodnosti) na cijenu
    # Proračun će biti računarski intezivan, aproksimacija vrhunskih feature-a
    if 'amenities_list' in all_df.columns:
        print("    - Analiza premijum uticaja specifičnih pogodnosti (Amenities)...")
        # Izravnavanje (flatten)
        sample = all_df.sample(min(20000, len(all_df)))
        all_amenities = pd.Series([a for sublist in sample['amenities_list'] for a in sublist])
        top_20 = all_amenities.value_counts().head(20).index.tolist()

        amenity_premium = []
        for am in top_20:
            has_am = sample['amenities_list'].apply(lambda x: am in x)
            price_with = sample.loc[has_am, 'price_usd'].mean()
            price_without = sample.loc[~has_am, 'price_usd'].mean()
            premium = price_with - price_without
            amenity_premium.append({'Amenity': am, 'Premium_USD': premium})

        premium_df = pd.DataFrame(amenity_premium).sort_values('Premium_USD', ascending=False)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=premium_df, y='Amenity', x='Premium_USD', hue='Amenity', palette='viridis', legend=False)
        plt.title('Koje pogodnosti (amenities) najviše podižu cijenu smještaja?')
        plt.xlabel('Premium (dodatna vrijednost u USD iznad prosjeka)')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_2_amenities_premium.png'), dpi=150)
        plt.close()

    print("    - Korelacija raspoloživih dana u godini i konačne cijene jedinice...")
    # Graf 8.3 — Dostupnost (availability_365) vs. cijena
    if 'availability_365' in all_df.columns:
        plt.figure(figsize=(14, 8))
        sns.scatterplot(data=all_df.sample(min(15000, len(all_df))), x='availability_365', y='price_usd',
                        hue='city', alpha=0.3, palette=CITY_COLORS)
        plt.yscale('log')
        plt.title('Godišnja kalendarska dostupnost vs. Cijena dionice (USD)')
        plt.xlabel('Dostupnost u narednih 365 dana')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_3_availability_vs_price.png'), dpi=150)
        plt.close()

    print("    - Rangiranje 15 najisplativijih kvartova...")
    # Graf 8.4 — Procijenjeni prihod po kvartu
    if 'estimated_revenue_l365d' in all_df.columns and 'neighbourhood_cleansed' in all_df.columns:
        rev_df = all_df.groupby(['city', 'neighbourhood_cleansed'])['estimated_revenue_l365d'].mean().reset_index()
        top_15_rev = rev_df.sort_values('estimated_revenue_l365d', ascending=False).head(15)

        plt.figure(figsize=(14, 8))
        sns.barplot(data=top_15_rev, y='neighbourhood_cleansed', x='estimated_revenue_l365d', hue='city', dodge=False)
        plt.title('Najisplativije geografske zone za obavljanje Airbnb poslovanja (L365D revenue)')
        plt.xlabel('Procijenjeni godišnji prihod')
        plt.ylabel('Administrativni kvart')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_4_revenue_by_neighborhood.png'), dpi=150)
        plt.close()

    print("    - Izdvajanje udijela arhitektonskih tipova nekretnina u ukupnoj ponudi...")
    # Graf 8.5 — Property type distribucija po gradu
    if 'property_type' in all_df.columns:
        top_types = all_df['property_type'].value_counts().head(8).index
        all_df['pt_clean'] = all_df['property_type'].where(all_df['property_type'].isin(top_types), 'Ostalo')

        pt_cross = pd.crosstab(all_df['city'], all_df['pt_clean'], normalize='index') * 100
        pt_cross.plot(kind='barh', stacked=True, figsize=(14, 8), colormap='tab20')
        plt.title('Arhitekturalni tipovi nekretnina na mikrolokalnim Airbnb tržištima')
        plt.xlabel('Udio nekretnina (%)')
        plt.legend(bbox_to_anchor=(1.05, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_5_property_type_distribution.png'), dpi=150)
        plt.close()

    print("    - Kreiranje proporcija legalnog (licenciranog) tržišta za svaku regiju...")
    # Graf 8.6 — Licence po gradu
    if 'license' in all_df.columns:
        all_df['has_license'] = all_df['license'].notna() & (all_df['license'] != '') & (all_df['license'] != 'Exempt')
        lic_summary = all_df.groupby('city')['has_license'].value_counts(normalize=True).unstack().fillna(0) * 100

        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for i, city in enumerate(lic_summary.index):
            axes[i].pie(lic_summary.loc[city], labels=['Bez licence/Izuzeti', 'Licencirano'],
                        autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'], wedgeprops=dict(width=0.4))
            axes[i].set_title(city)
        plt.suptitle('Regulatorna usklađenost Airbnb tržišta — postotak licenciranog poslovanja', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_6_license_donuts.png'), dpi=150)
        plt.close()

    print("    - Plotanje linije historijskog rasta inventara na vremenskoj skali...")
    # Graf 8.7 — Rast tržišta (sezonalnost)
    if 'first_review' in all_df.columns:
        all_df['first_review_year'] = pd.to_datetime(all_df['first_review']).dt.year
        valid_years = all_df[(all_df['first_review_year'] >= 2010) & (all_df['first_review_year'] <= 2026)]

        growth = valid_years.groupby(['first_review_year', 'city']).size().unstack().fillna(0)

        plt.figure(figsize=(14, 8))
        growth.plot(linewidth=2, figsize=(14, 8), color=[CITY_COLORS.get(x, '#333333') for x in growth.columns])
        plt.title('Istorijski rast kapaciteta Airbnb tržišta — Prva recenzija po godini platformizacije')
        plt.xlabel('Godina (Year)')
        plt.ylabel('Broj novih nekretnina (Aktivacija)')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_7_market_growth.png'), dpi=150)
        plt.close()

    print("    - Analiza zahtijevanih dužina (minimum noćenja) - tranzicija ka dugoročnom iznajmljivanju...")
    # Graf 8.8 — Minimum nights
    if 'minimum_nights' in all_df.columns:
        plt.figure(figsize=(14, 8))
        hist_df = all_df[all_df['minimum_nights'] <= 30]
        sns.histplot(data=hist_df, x='minimum_nights', hue='city', multiple='stack', palette=CITY_COLORS, bins=30)
        plt.title('Minimalan broj noćenja — Tranzicija od turističkog prema dugoročnom stambenom najmu iznad 30 dana')
        plt.xlabel('Propisan minimum u noćenjima')
        plt.xticks([1, 2, 7, 14, 30])
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_8_minimum_nights.png'), dpi=150)
        plt.close()
