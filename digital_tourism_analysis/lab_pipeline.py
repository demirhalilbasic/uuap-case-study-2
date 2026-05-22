import os
import sys

from src.data_loader import load_all_cities
from src.price_analysis import run_price_analysis, run_cross_city_comparison
from src.geo_analysis import run_geo_analysis
from src.hypothesis_testing import run_hypothesis_testing
from src.host_analysis import run_host_analysis
from src.review_analysis import run_review_analysis
from src.advanced_analysis import run_advanced_analysis

def run_full_pipeline():
    print("==================================================================")
    print("      UuAP Case Study 2 - Analiza Digitalnog Turizma (Airbnb)     ")
    print("      Autori: Demir Halilbašić, Belma Salkičić, Ajnur Nukić       ")
    print("      IPI Academy Tuzla | Spring 2026                             ")
    print("==================================================================")

    print("\nKorak 1/8: Učitavanje i preprocesiranje podataka...")
    city_dfs, all_df = load_all_cities()

    if all_df.empty:
        print("Dataset je prazan. Izvršavanje zaustavljeno.")
        sys.exit(1)

    print(f"\nUčitano ukupno opservacija: {len(all_df)} u {len(city_dfs)} gradova.")

    print("\nKorak 2/8: Analiza distribucije cijena...")
    run_price_analysis(city_dfs, all_df)

    print("\nKorak 3/8: Geografske analize...")
    run_geo_analysis(city_dfs, all_df)

    print("\nKorak 4/8: Analiza po tipu smještaja i komparacija gradova...")
    run_cross_city_comparison(all_df)

    print("\nKorak 5/8: Testiranje hipoteza...")
    run_hypothesis_testing(city_dfs, all_df)

    print("\nKorak 6/8: Analiza hostova...")
    run_host_analysis(city_dfs, all_df)

    print("\nKorak 7/8: Analiza recenzija i ocjena...")
    run_review_analysis(city_dfs, all_df)

    print("\nKorak 8/8: Napredne analize (klasterizacija, procjena prihoda)...")
    run_advanced_analysis(city_dfs, all_df)

    print("\nAnalitički proces (Pipeline) uredno završen.")
    print("Svi generisani grafici i tekstualni izvještaji su sačuvani unutar odgovarajućih Output direktorija.")

if __name__ == "__main__":
    run_full_pipeline()
