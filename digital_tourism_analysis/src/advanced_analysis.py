"""
advanced_analysis.py
--------------------
Napredne analitičke metode: KMeans klasterizacija, amenities premium,
analiza dostupnosti, procijenjeni prihod, tipovi nekretnina, licence i rast tržišta.

FIX: Graf 8.7 — growth.plot() vraća axes objekt, ne figure.
     Sada se eksplicitno kreira fig objekat i sprema fig.savefig(),
     što garantuje da se sprema tačno taj grafik.
"""
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from src.utils import CITY_COLORS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'


def run_advanced_analysis(city_dfs, all_df):
    out_dir = 'output/08_advanced_analysis'
    ensure_dir_exists(out_dir)

    # ------------------------------------------------------------------
    # Graf 8.1 — KMeans klasterizacija tržišnih segmenata
    # ------------------------------------------------------------------
    print("    - Konfiguracija i pokretanje K-Means klasterizacije...")
    cluster_features = [
        'price_usd', 'distance_from_center',
        'review_scores_rating', 'amenity_count', 'accommodates'
    ]
    if all(f in all_df.columns for f in cluster_features):
        cluster_df = all_df[cluster_features + ['city']].dropna().copy()

        if len(cluster_df) > 50:
            scaler = StandardScaler()
            scaled = scaler.fit_transform(cluster_df[cluster_features])

            # Elbow metoda
            inertias = []
            k_range = range(2, 9)
            for k in k_range:
                km = KMeans(n_clusters=k, random_state=42, n_init=10)
                km.fit(scaled)
                inertias.append(km.inertia_)

            fig, axes = plt.subplots(1, 2, figsize=(18, 7))

            axes[0].plot(list(k_range), inertias, 'bo-', linewidth=2, markersize=8)
            axes[0].set_title('Elbow metoda — optimalan broj klastera')
            axes[0].set_xlabel('Broj klastera (K)')
            axes[0].set_ylabel('Inertia')
            axes[0].grid(True)

            # Finalni model K=4
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            cluster_df['Klaster'] = kmeans.fit_predict(scaled).astype(str)

            sample_c = cluster_df.sample(min(15000, len(cluster_df)), random_state=42)
            sns.scatterplot(
                data=sample_c,
                x='distance_from_center', y='price_usd',
                hue='Klaster', palette='Set1', alpha=0.5, ax=axes[1]
            )
            axes[1].set_yscale('log')
            axes[1].set_title('KMeans klasteri (K=4) — segmenti Airbnb tržišta')
            axes[1].set_xlabel('Udaljenost od centra (km)')
            axes[1].set_ylabel('Cijena (USD, log)')

            plt.suptitle('KMeans klasterizacija tržišnih segmenata Airbnb platforme',
                         fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, '8_1_kmeans_clustering.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Graf 8.2 — Amenities premium (top 20 pogodnosti i uticaj na cijenu)
    # ------------------------------------------------------------------
    print("    - Evaluacija premijum uticaja pogodnosti (Amenities)...")
    if 'amenities_list' in all_df.columns:
        sample = all_df.sample(min(20000, len(all_df)), random_state=42)
        all_amenities = pd.Series(
            [a for sublist in sample['amenities_list'] for a in sublist]
        )
        top_20 = all_amenities.value_counts().head(20).index.tolist()

        premium_rows = []
        for am in top_20:
            has_am = sample['amenities_list'].apply(lambda x: am in x)
            price_with    = sample.loc[has_am,  'price_usd'].mean()
            price_without = sample.loc[~has_am, 'price_usd'].mean()
            if pd.notna(price_with) and pd.notna(price_without):
                premium_rows.append({
                    'Pogodnost': am,
                    'Premium_USD': price_with - price_without
                })

        if premium_rows:
            prem_df = pd.DataFrame(premium_rows).sort_values('Premium_USD', ascending=True)
            colors = ['#d62728' if v > 0 else '#1f77b4' for v in prem_df['Premium_USD']]

            plt.figure(figsize=(14, 9))
            plt.barh(prem_df['Pogodnost'], prem_df['Premium_USD'], color=colors)
            plt.axvline(0, color='black', linewidth=1)
            plt.title('Koje pogodnosti (amenities) najviše podižu cijenu smještaja?',
                      fontsize=14)
            plt.xlabel('Prosječni premium (USD) u odnosu na listinge bez te pogodnosti')
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, '8_2_amenities_premium.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()

    # ------------------------------------------------------------------
    # Graf 8.3 — Dostupnost (availability_365) vs. cijena
    # ------------------------------------------------------------------
    print("    - Korelacija raspoloživih dana i konačne cijene...")
    if 'availability_365' in all_df.columns:
        sample = all_df.sample(min(15000, len(all_df)), random_state=42)
        plt.figure(figsize=(14, 8))
        sns.scatterplot(
            data=sample, x='availability_365', y='price_usd',
            hue='city', alpha=0.3, palette=CITY_COLORS, s=20
        )
        plt.yscale('log')
        plt.title('Godišnja dostupnost (dani) vs. Cijena (USD)\n'
                  'Što je dostupnost niža, je li listing popularniji i skuplji?',
                  fontsize=13)
        plt.xlabel('Broj dostupnih dana u narednih 365 dana')
        plt.ylabel('Cijena po noćenju (USD, log)')
        plt.legend(title='Grad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_3_availability_vs_price.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.4 — Top 15 najisplativijih kvartova po prihodu
    # ------------------------------------------------------------------
    print("    - Rangiranje 15 najisplativijih kvartova...")
    if 'estimated_revenue_l365d' in all_df.columns and \
       'neighbourhood_cleansed' in all_df.columns:
        rev_df = (all_df
                  .groupby(['city', 'neighbourhood_cleansed'])['estimated_revenue_l365d']
                  .mean()
                  .reset_index())
        top15 = rev_df.sort_values('estimated_revenue_l365d', ascending=False).head(15)
        top15['Kvart'] = top15['neighbourhood_cleansed'] + ' (' + top15['city'] + ')'

        plt.figure(figsize=(14, 8))
        sns.barplot(
            data=top15, y='Kvart', x='estimated_revenue_l365d',
            hue='city', dodge=False, palette=CITY_COLORS
        )
        plt.title('Najisplativiji kvartovi za Airbnb poslovanje (procijenjeni godišnji prihod)',
                  fontsize=13)
        plt.xlabel('Procijenjeni prosječni godišnji prihod (lokalna valuta)')
        plt.ylabel('')
        plt.legend(title='Grad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_4_revenue_by_neighborhood.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.5 — Property type distribucija po gradu (stacked horizontal bar)
    # ------------------------------------------------------------------
    print("    - Izdvajanje tipova nekretnina u ukupnoj ponudi...")
    if 'property_type' in all_df.columns:
        top_types = all_df['property_type'].value_counts().head(8).index
        df_pt = all_df.copy()
        df_pt['Tip nekretnine'] = df_pt['property_type'].where(
            df_pt['property_type'].isin(top_types), 'Ostalo'
        )
        pt_cross = (pd.crosstab(df_pt['city'], df_pt['Tip nekretnine'], normalize='index')
                    * 100)

        ax = pt_cross.plot(kind='barh', stacked=True, figsize=(16, 8), colormap='tab20')
        plt.title('Arhitektonski tipovi nekretnina na Airbnb tržištu po gradu', fontsize=14)
        plt.xlabel('Udio (%)')
        plt.ylabel('Grad')
        plt.legend(title='Tip nekretnine', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_5_property_type_distribution.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.6 — Licence (regulatorna usklađenost) — donut chart po gradu
    # ------------------------------------------------------------------
    print("    - Kreiranje proporcija licenciranog tržišta po gradu...")
    if 'license' in all_df.columns:
        all_df['has_license'] = (
            all_df['license'].notna() &
            (all_df['license'].astype(str).str.strip() != '') &
            (all_df['license'].astype(str).str.lower() != 'exempt')
        )
        lic_summary = (all_df.groupby('city')['has_license']
                       .value_counts(normalize=True)
                       .unstack()
                       .fillna(0) * 100)

        cities = list(lic_summary.index)
        n = len(cities)
        ncols = 3
        nrows = (n + ncols - 1) // ncols

        fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5 * nrows))
        axes = axes.flatten()

        i = -1
        for i, city in enumerate(cities):
            vals = lic_summary.loc[city]
            labels = ['Bez licence', 'Licencirano']
            colors = ['#ff9999', '#66b3ff']
            axes[i].pie(
                vals, labels=labels, autopct='%1.1f%%',
                colors=colors, wedgeprops=dict(width=0.45),
                startangle=90
            )
            axes[i].set_title(city, fontsize=12)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        plt.suptitle(
            'Regulatorna usklađenost Airbnb tržišta — postotak licenciranog poslovanja',
            fontsize=14
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_6_license_donuts.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.7 — Historijski rast tržišta (broj novih listinga po godini)
    # FIX: growth.plot() vraća axes, ne figure — eksplicitno dohvatamo fig
    # ------------------------------------------------------------------
    print("    - Plotanje historijskog rasta inventara na vremenskoj skali...")
    if 'first_review' in all_df.columns:
        all_df['first_review_year'] = pd.to_datetime(
            all_df['first_review'], errors='coerce'
        ).dt.year

        valid = all_df[
            (all_df['first_review_year'] >= 2010) &
            (all_df['first_review_year'] <= 2026)
        ]
        growth = valid.groupby(['first_review_year', 'city']).size().unstack().fillna(0)

        # FIX: eksplicitno kreiramo fig i ax, pa koristimo fig.savefig()
        fig, ax = plt.subplots(figsize=(14, 8))
        for city in growth.columns:
            ax.plot(
                growth.index, growth[city],
                linewidth=2.5, label=city,
                color=CITY_COLORS.get(city, '#333333'),
                marker='o', markersize=5
            )

        ax.set_title(
            'Historijski rast kapaciteta Airbnb tržišta po gradu\n'
            '(Broj listinga s prvom recenzijom po godini)',
            fontsize=14
        )
        ax.set_xlabel('Godina')
        ax.set_ylabel('Broj novih listinga')
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(title='Grad')
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, '8_7_market_growth.png'),
                    dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------
    # GRAF 8.8 — Minimum nights distribucija
    # ------------------------------------------------------------------
    print("    - Analiza zahtijevanih minimuma noćenja...")
    if 'minimum_nights' in all_df.columns:
        hist_df = all_df[
            (all_df['minimum_nights'] >= 1) &
            (all_df['minimum_nights'] <= 30)
        ]
        plt.figure(figsize=(14, 8))
        sns.histplot(
            data=hist_df, x='minimum_nights', hue='city',
            multiple='stack', palette=CITY_COLORS, bins=30
        )
        plt.axvline(1,  color='green',  linestyle='--', alpha=0.7, label='1 noć (turistički)')
        plt.axvline(7,  color='orange', linestyle='--', alpha=0.7, label='7 noći (sedmični)')
        plt.axvline(30, color='red',    linestyle='--', alpha=0.7, label='30 noći (dugoročni)')
        plt.title(
            'Minimalan broj noćenja — turistički vs. dugoročni najam',
            fontsize=14
        )
        plt.xlabel('Minimalan broj noćenja')
        plt.xticks([1, 2, 3, 5, 7, 14, 21, 30])
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_8_minimum_nights.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.9 — Value-for-Money scatter: ocjena vs cijena (kvadranti)
    # ------------------------------------------------------------------
    print("    - Generisanje Value-for-Money scatter grafa (ocjena vs. cijena)...")
    if 'price_usd' in all_df.columns and 'review_scores_rating' in all_df.columns:
        vfm_sample = all_df[
            all_df['price_usd'].notna() & all_df['review_scores_rating'].notna()
        ].sample(min(20000, len(all_df)), random_state=42)

        plt.figure(figsize=(14, 10))
        sns.scatterplot(
            data=vfm_sample, x='review_scores_rating', y='price_usd',
            hue='city', palette=CITY_COLORS, alpha=0.5, s=30
        )
        plt.yscale('log')
        plt.xlabel('Prosječna ocjena gostiju')
        plt.ylabel('Cijena po noćenju (USD, log skala)')
        plt.title('Value-for-Money: Ocjena vs. Cijena — traženje "sweet spot" ponuda')

        # Add median lines to create quadrants
        median_price = vfm_sample['price_usd'].median()
        median_rating = vfm_sample['review_scores_rating'].median()
        plt.axhline(median_price, color='grey', linestyle='--', linewidth=1)
        plt.axvline(median_rating, color='grey', linestyle='--', linewidth=1)

        # Annotate a few representative "value" listings: high rating, low price
        try:
            low_price_thresh = vfm_sample['price_usd'].quantile(0.25)
            high_rating_thresh = vfm_sample['review_scores_rating'].quantile(0.75)
            candidates = vfm_sample[
                (vfm_sample['price_usd'] <= low_price_thresh) &
                (vfm_sample['review_scores_rating'] >= high_rating_thresh)
            ]
            for _, r in candidates.sort_values(['review_scores_rating', 'price_usd'],
                                              ascending=[False, True]).head(6).iterrows():
                label = f"{r.get('neighbourhood_cleansed', '')}\n${r['price_usd']:.0f}"
                plt.annotate(label, (r['review_scores_rating'], r['price_usd']),
                             textcoords="offset points", xytext=(6, -6), fontsize=8)
        except Exception:
            pass

        plt.legend(title='Grad', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '8_9_value_for_money_scatter.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ------------------------------------------------------------------
    # Graf 8.10 — Udaljenost od centra vs. cijena (hexbin / 2D density)
    # ------------------------------------------------------------------
    print("    - Generisanje hexbin grafa: udaljenost od centra vs. cijena...")
    if 'distance_from_center' in all_df.columns and 'price_usd' in all_df.columns:
        hex_df = all_df[
            all_df['distance_from_center'].notna() & all_df['price_usd'].notna()
        ].copy()
        if not hex_df.empty:
            plt.figure(figsize=(14, 8))
            # Use matplotlib hexbin for density; log y helps with skew
            hb = plt.hexbin(hex_df['distance_from_center'], hex_df['price_usd'],
                            gridsize=60, cmap='viridis', bins='log')
            plt.yscale('log')
            plt.colorbar(hb, label='log(N)')
            plt.xlabel('Udaljenost od centra (km)')
            plt.ylabel('Cijena po noćenju (USD, log skala)')
            plt.title('Gustoća listinga: Udaljenost od centra vs. Cijena (hexbin)')
            # Trendline (robust lowess via seaborn regplot on small sample)
            try:
                sample = hex_df.sample(min(10000, len(hex_df)), random_state=42)
                sns.regplot(data=sample, x='distance_from_center', y='price_usd',
                            scatter=False, lowess=True, color='white', line_kws={'linewidth':1.5, 'alpha':0.9})
            except Exception:
                pass

            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, '8_10_distance_vs_price_hexbin.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()