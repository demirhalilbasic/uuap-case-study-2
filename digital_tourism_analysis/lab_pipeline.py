import os
import sys
import time

# VAŽNO: mora biti prije svih matplotlib importa
import matplotlib
matplotlib.use('Agg')

from src.data_loader import load_all_cities
from src.data_cleaner import clean_all_cities
from src.price_analysis import run_price_analysis, run_cross_city_comparison
from src.geo_analysis import run_geo_analysis
from src.hypothesis_testing import run_hypothesis_testing
from src.host_analysis import run_host_analysis
from src.review_analysis import run_review_analysis
from src.advanced_analysis import run_advanced_analysis


def run_full_pipeline():
    start_time = time.time()

    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║   UuAP Case Study 2 — Analiza Digitalnog Turizma (Airbnb)        ║")
    print("║   Autori: Demir Halilbašić, Belma Salkičić, Ajnur Nukić          ║")
    print("║   IPI Academy Tuzla | Spring 2026                                ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    # -------------------------------------------------------------------
    # KORAK 1: Učitavanje sirovih CSV fajlova
    # -------------------------------------------------------------------
    print("\nKorak 1/9: Učitavanje sirovih podataka...")
    raw_city_dfs, _ = load_all_cities()

    if not raw_city_dfs:
        print("Greška: Nisu pronađeni fajlovi. Pipeline zaustavljen.")
        sys.exit(1)

    # -------------------------------------------------------------------
    # KORAK 2: Čišćenje i feature engineering (odvojeni modul)
    # -------------------------------------------------------------------
    print("\nKorak 2/9: Čišćenje i preprocesiranje podataka...")
    city_dfs, all_df = clean_all_cities(raw_city_dfs)

    if all_df.empty:
        print("Greška: Dataset je prazan nakon čišćenja. Pipeline zaustavljen.")
        sys.exit(1)

    print(f"\nUkupno čistih opservacija: {len(all_df):,} u {len(city_dfs)} gradova")

    # -------------------------------------------------------------------
    # KORACI 3–9: Analiza
    # -------------------------------------------------------------------
    print("\nKorak 3/9: Analiza distribucije cijena...")
    run_price_analysis(city_dfs, all_df)

    print("\nKorak 4/9: Geografske analize i mape...")
    run_geo_analysis(city_dfs, all_df)

    print("\nKorak 5/9: Komparacija između gradova...")
    run_cross_city_comparison(all_df)

    print("\nKorak 6/9: Testiranje hipoteze (OLS, ANOVA, Kruskal-Wallis)...")
    run_hypothesis_testing(city_dfs, all_df)

    print("\nKorak 7/9: Analiza hostova...")
    run_host_analysis(city_dfs, all_df)

    print("\nKorak 8/9: Analiza recenzija i ocjena...")
    run_review_analysis(city_dfs, all_df)

    print("\nKorak 9/9: Napredne analize (klasterizacija, amenities, prihod)...")
    run_advanced_analysis(city_dfs, all_df)

    elapsed = time.time() - start_time
    print(f"\nPipeline završen! Ukupno vrijeme izvršavanja: {elapsed:.1f}s")
    print("Svi grafici sačuvani u odgovarajućim output/ poddirektorijima.")


if __name__ == "__main__":
    run_full_pipeline()
