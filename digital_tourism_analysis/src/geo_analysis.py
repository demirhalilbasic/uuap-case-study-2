import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import folium
from folium.plugins import HeatMap
from scipy import stats
import json
from src.utils import CITY_COLORS, CITY_FILES, CITY_CENTERS, ensure_dir_exists

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'

def run_geo_analysis(city_dfs, all_df):
    out_dir = 'output/02_geo_analysis'

    out_dir_maps = os.path.join(out_dir, 'scatter_maps')
    out_dir_folium = os.path.join(out_dir, 'folium_maps')
    out_dir_hoods = os.path.join(out_dir, 'neighborhoods')
    out_dir_dist = os.path.join(out_dir, 'distance_analysis')

    ensure_dir_exists(out_dir_maps)
    ensure_dir_exists(out_dir_folium)
    ensure_dir_exists(out_dir_hoods)
    ensure_dir_exists(out_dir_dist)

    print("    - Kreiranje scatter mapa na bazi koordinata (Plotly HTML)...")
    # Graf 2.1 — Scatter mapa lokacija listinga
    # This will be simpler to do separate html files per city using plotly due to massive size
    for city, df in city_dfs.items():
        if 'latitude' in df.columns and 'longitude' in df.columns:
            # We sample if too large to prevent browser crash
            sample_df = df.sample(n=min(5000, len(df)), random_state=42)
            fig = px.scatter_mapbox(sample_df, lat="latitude", lon="longitude", hover_name="name",
                                    hover_data=["price_usd", "neighbourhood_cleansed", "room_type"],
                                    color="price_usd", size=sample_df["reviews_per_month"].fillna(0.1),
                                    color_continuous_scale="RdYlGn_r", size_max=15, zoom=10,
                                    mapbox_style="carto-positron")
            fig.update_layout(title=f"Geografska distribucija listinga i cijena — {city}")
            fig.write_html(os.path.join(out_dir_maps, f'2_1_scatter_map_{city}.html'))

    print("    - Računanje i iscrtavanje korelacijskih odnosa udaljenosti od centra (Pearson test)...")
    # Graf 2.3 — Udaljenost od centra vs. cijena (scatter + regresijska linija)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    for i, city in enumerate(city_dfs.keys()):
        df = city_dfs[city]
        sns.regplot(data=df, x='distance_from_center', y='price_usd', ax=axes[i],
                    scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'}, lowess=True)
        clean_data = df[['distance_from_center', 'price_usd']].dropna()
        if len(clean_data) > 1:
            r, p = stats.pearsonr(clean_data['distance_from_center'], clean_data['price_usd'])
            axes[i].set_title(f"{city} (r={r:.3f}, p={p:.3e})")
        else:
            axes[i].set_title(f"{city} (Nema dovoljno podataka)")
        axes[i].set_xlabel('Udaljenost od centra (km)')
        axes[i].set_ylabel('Cijena (USD)')
    plt.suptitle('Udaljenost od centra vs. cijena — postoji li korelacija?', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_dist, '2_3_distance_vs_price.png'), dpi=150, bbox_inches='tight')
    plt.close()

    print("    - Pokretanje 3D mapiranja geografskih i cjenovnih anomalija...")
    # Graf 2.5 — 3D scatter plot (Plotly): lat × long × price
    for city, df in city_dfs.items():
        if 'latitude' in df.columns and 'longitude' in df.columns:
            sample_df = df.sample(n=min(3000, len(df)), random_state=42)
            fig = px.scatter_3d(sample_df, x='longitude', y='latitude', z='price_usd',
                                color='room_type', size_max=10, opacity=0.7)
            fig.update_layout(title=f"3D prikaz cijena po geografskoj lokaciji — {city}")
            fig.write_html(os.path.join(out_dir_maps, f'2_5_3d_scatter_{city}.html'))

    print("    - Sortiranje 10 najskupljih i najjeftinijih kvartova...")
    # Graf 2.6 — Top 10 najskupljih i top 10 najjeftinijih kvartova
    for city, df in city_dfs.items():
        if 'neighbourhood_cleansed' in df.columns:
            hood_prices = df.groupby('neighbourhood_cleansed')['price_usd'].mean().dropna().sort_values()
            if len(hood_prices) == 0:
                continue

            n_cheap = min(10, len(hood_prices))
            n_expensive = min(10, len(hood_prices) - n_cheap) if len(hood_prices) > n_cheap else 0

            top_cheap = hood_prices.head(n_cheap)
            top_expensive = hood_prices.tail(n_expensive) if n_expensive > 0 else pd.Series(dtype=float)

            plt.figure(figsize=(14, 8))
            all_hoods = pd.concat([top_cheap, top_expensive])

            if not all_hoods.empty:
                colors = ['green']*len(top_cheap) + ['red']*len(top_expensive)
                all_hoods.plot(kind='barh', color=colors)
                plt.title(f'Najskuplji i najjeftiniji kvartovi — {city}')
                plt.xlabel('Prosječna cijena (USD)')
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir_hoods, f'2_6_top_bottom_hoods_{city}.png'), dpi=150, bbox_inches='tight')
            plt.close()

    print("    - Inicijalizacija i preklapanje Folium interaktivnih GeoJSON mapa...")
    # Folium mape (2.2 i 2.4)
    for city, file_prefix in CITY_FILES.items():
        df = city_dfs[city]
        geojson_path = os.path.join('data/neighbourhoods', f"{file_prefix}.geojson")

        if os.path.exists(geojson_path) and 'neighbourhood_cleansed' in df.columns:
            print(f"      Započinjem mapiranje opština za: {city}...")
            with open(geojson_path, 'r', encoding='utf-8') as f:
                geo_data = json.load(f)

            avg_price_df = df.groupby('neighbourhood_cleansed')['price_usd'].mean().reset_index()
            count_df = df.groupby('neighbourhood_cleansed').size().reset_index(name='count')

            lat_mean = df['latitude'].mean()
            lon_mean = df['longitude'].mean()

            if pd.isna(lat_mean) or pd.isna(lon_mean):
                lat_mean, lon_mean = CITY_CENTERS[city]

            # Heatmapa cijena
            m_price = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)
            folium.Choropleth(
                geo_data=geo_data,
                name='choropleth',
                data=avg_price_df,
                columns=['neighbourhood_cleansed', 'price_usd'],
                key_on='feature.properties.neighbourhood',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Prosječna cijena (USD)'
            ).add_to(m_price)
            folium.LayerControl().add_to(m_price)
            m_price.save(os.path.join(out_dir_folium, f'2_2_{CITY_FILES[city]}_heatmap.html'))

            # Choropleth gustoca
            m_count = folium.Map(location=[lat_mean, lon_mean], zoom_start=11)
            folium.Choropleth(
                geo_data=geo_data,
                name='choropleth',
                data=count_df,
                columns=['neighbourhood_cleansed', 'count'],
                key_on='feature.properties.neighbourhood',
                fill_color='Blues',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name='Broj listinga'
            ).add_to(m_count)
            folium.LayerControl().add_to(m_count)
            m_count.save(os.path.join(out_dir_folium, f'2_4_{CITY_FILES[city]}_density.html'))
