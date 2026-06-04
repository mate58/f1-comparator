"""
etl_qualifying.py
=================
Descarga la clasificacion (Qualifying) de Jolpi API y la sube a Supabase.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

SLEEP_NORMAL = 1.0
SLEEP_429    = 60.0
MAX_RETRIES  = 5


def get_with_retry(url: str, params: dict) -> dict | None:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 429:
                wait = SLEEP_429 * attempt
                print(f"\n  Rate limit (429) — esperando {wait:.0f}s (intento {attempt}/{MAX_RETRIES})...")
                time.sleep(wait)
                continue
            print(f"\n  Error HTTP {resp.status_code} — intento {attempt}/{MAX_RETRIES}")
            time.sleep(2 ** attempt)
        except requests.exceptions.RequestException as e:
            print(f"\n  Error de red: {e} — intento {attempt}/{MAX_RETRIES}")
            time.sleep(2 ** attempt)
    print(f"\n  Agotados {MAX_RETRIES} reintentos. Guardando progreso hasta aqui.")
    return None


def etl_qualifying():
    print("Descargando Qualifying de Jolpi API...")
    base_url = "http://api.jolpi.ca/ergast/f1/qualifying.json"

    limit    = 100
    offset   = 0
    total    = 1      # se actualiza con el primer response
    all_data = []

    while offset < total:
        data = get_with_retry(base_url, {"limit": limit, "offset": offset})

        if data is None:
            print(f"  Descarga interrumpida en offset {offset}. "
                  f"Guardando {len(all_data)} registros obtenidos.")
            break

        mrdata = data.get("MRData", {})
        total  = int(mrdata.get("total", 0))  # total de FILAS de qualifying, no de carreras
        races  = mrdata.get("RaceTable", {}).get("Races", [])

        if not races:
            break

        for race in races:
            race_id = f"{race.get('season')}_{race.get('round')}"
            for q in race.get("QualifyingResults", []):
                all_data.append({
                    "race_id"       : race_id,
                    "driver_id"     : q.get("Driver", {}).get("driverId"),
                    "constructor_id": q.get("Constructor", {}).get("constructorId"),
                    "position"      : str(q.get("position")),
                    "q1"            : q.get("Q1") or None,
                    "q2"            : q.get("Q2") or None,
                    "q3"            : q.get("Q3") or None,
                })

        # FIX: avanzar siempre por `limit`, no por len(races)
        # La API pagina por filas de qualifying, no por carreras.
        # len(races)=5 carreras != 100 filas de qualifying
        offset += limit

        pct = min(offset / total * 100, 100) if total else 0
        print(f"  {len(all_data):>6} registros descargados ({pct:.1f}%)", end="\r")
        time.sleep(SLEEP_NORMAL)

    print(f"\n  Descarga completada: {len(all_data)} registros.")

    if not all_data:
        print("  Sin datos para cargar.")
        return

    # Subir a Supabase
    print(f"Subiendo a Supabase (tabla 'qualifying')...")
    df = pd.DataFrame(all_data)
    
    # --- LIMPIEZA AGRESIVA ---
    # Convertimos el DataFrame a un formato compatible con JSON
    # Reemplazamos todos los NaN por None explícitamente
    import numpy as np
    df = df.replace({np.nan: None})
    records = df.to_dict("records")
    # --------------------------

    total_s = 0
    for i in range(0, len(records), 500):
        batch = records[i:i+500]
        try:
            supabase.table("qualifying").upsert(batch).execute()
            total_s += len(batch)
            print(f"  {total_s}/{len(records)} subidos...", end="\r")
        except Exception as e:
            print(f"\n  Error en lote {i//500 + 1}: {e}")
            # Esto imprimirá el error exacto para ver qué columna está molestando


if __name__ == "__main__":
    etl_qualifying()