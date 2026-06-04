import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# CONFIGURACIÓN
# ==========================================
load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

BASE_URL = "https://api.jolpi.ca/ergast/f1"

# El campeonato de constructores empezó en 1958
SEASON_START = 1958
SEASON_END   = 2026


# ==========================================
# ETL PRINCIPAL
# ==========================================
def etl_constructor_championship():
    """
    Descarga el campeonato de constructores histórico desde jolpica-f1
    iterando por temporada (el endpoint global sin {season} no existe).

    Produce las columnas que espera elo.py:
        - championship_id  → PK compuesta "year_constructor_id"
        - year             (int)
        - constructor_id   (str)
        - position         (int)
    """
    print("⏳ Descargando campeonato de constructores...")

    all_records = []

    for season in range(SEASON_START, SEASON_END + 1):
        url = f"{BASE_URL}/{season}/constructorstandings.json?limit=100"
        response = requests.get(url)

        if response.status_code == 404:
            # Temporada sin datos (antes del campeonato de constructores)
            continue

        if response.status_code != 200:
            print(f"  ⚠️  HTTP {response.status_code} en {season} — saltando")
            continue

        standings_lists = (
            response.json()
            .get("MRData", {})
            .get("StandingsTable", {})
            .get("StandingsLists", [])
        )

        if not standings_lists:
            continue

        # Tomamos el último StandingsList de la temporada (standings finales)
        final_standings = standings_lists[-1]

        for entry in final_standings.get("ConstructorStandings", []):
            constructor_id = str(entry.get("Constructor", {}).get("constructorId", ""))
            all_records.append({
                "championship_id": f"{season}_{constructor_id}",
                "year":            season,
                "constructor_id":  constructor_id,
                "position":        int(entry.get("position", 0)),
            })

        print(f"  ✓ {season}: {len(final_standings.get('ConstructorStandings', []))} escuderías")

    # ==========================================
    # TRANSFORMACIÓN FINAL
    # ==========================================
    if not all_records:
        print("⚠️  No se encontraron datos para cargar.")
        return

    df = pd.DataFrame(all_records)
    df = df.dropna(subset=["year", "constructor_id", "position"])
    df = df.replace({np.nan: None})

    print(f"\n📊 Total registros: {len(df)}")
    print(f"   Temporadas: {df['year'].min()} → {df['year'].max()}")
    print(f"   Escuderías únicas: {df['constructor_id'].nunique()}")

    # ==========================================
    # CARGA A SUPABASE (UPSERT)
    # ==========================================
    records_to_upload = df.to_dict(orient="records")

    try:
        supabase.table("constructor_championship").upsert(records_to_upload).execute()
        print(f"\n✅ ¡{len(records_to_upload)} registros cargados en 'constructor_championship'!")
    except Exception as e:
        print(f"\n❌ Error en la subida: {e}")


# ==========================================
# ENTRY POINT
# ==========================================
if __name__ == "__main__":
    etl_constructor_championship()