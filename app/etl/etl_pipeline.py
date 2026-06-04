"""
etl_pipeline.py
===============
Extrae resultados históricos de carreras puramente desde la Jolpi API 
y los carga directamente en la tabla 'results' de Supabase.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

JOLPI_BASE_URL = "http://api.jolpi.ca/ergast/f1/results.json"

# ──────────────────────────────────────────────
# 1. EXTRACT (Solo Jolpi)
# ──────────────────────────────────────────────

def extract_jolpi_results() -> pd.DataFrame:
    print("⏳ [EXTRACT] Descargando historial oficial de Jolpi F1 API...")
    limit = 100 
    offset = 0
    total = 1 
    all_results = []
    
    while offset < total:
        url = f"{JOLPI_BASE_URL}?limit={limit}&offset={offset}"
        response = requests.get(url)
        
        # 🚨 FIX: Manejo inteligente del error 429
        if response.status_code == 429:
            print("⏳ API bloqueada (429). Esperando 30 segundos antes de reintentar...")
            time.sleep(30)
            continue # Reintenta la misma página
            
        if response.status_code != 200:
            print(f"❌ Error crítico: {response.status_code}")
            break
            
        data = response.json()
        mrdata = data.get("MRData", {})
        total = int(mrdata.get("total", 0))
        races = mrdata.get("RaceTable", {}).get("Races", [])
        
        resultados_en_esta_pagina = 0
        for race in races:
            for result in race.get("Results", []):
                all_results.append({
                    "race_id": f"{race.get('season')}_{race.get('round')}",
                    "year": int(race.get("season")),
                    "round": int(race.get("round")),
                    "driver_id": result.get("Driver", {}).get("driverId"),
                    "constructor_id": result.get("Constructor", {}).get("constructorId"),
                    "grid": result.get("grid"),
                    "position": str(result.get("position")),
                    "positionOrder": str(result.get("positionText")),
                    "status": result.get("status")
                })
                resultados_en_esta_pagina += 1
                
        print(f"  → Procesados {len(all_results)} resultados de {total}...")
        
        if resultados_en_esta_pagina == 0: break
        offset += resultados_en_esta_pagina
        time.sleep(2) # 🚨 Aumentamos la pausa de 0.5s a 2s para ser más amigables con la API
        
    return pd.DataFrame(all_results)

# ──────────────────────────────────────────────
# 2. TRANSFORM
# ──────────────────────────────────────────────

def transform_data(df_results: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ [TRANSFORM] Limpiando y estandarizando datos...")
    df_res = df_results.copy()
    
    # Estandarizamos IDs
    df_res["race_id"] = df_res["race_id"].astype(str).str.strip()
    df_res["driver_id"] = df_res["driver_id"].astype(str).str.strip().str.lower()
    df_res["constructor_id"] = df_res["constructor_id"].astype(str).str.strip().str.lower()
    
    # 🚨 FIX: Ya no forzamos a numérico 'position' y 'positionOrder'.
    # Los dejamos como texto para que 'R', 'W', 'D' se guarden correctamente.
    df_res["position"] = df_res["position"].astype(str)
    df_res["positionOrder"] = df_res["positionOrder"].astype(str)
    
    # Grid sí es numérico siempre, así que lo mantenemos
    df_res["grid"] = pd.to_numeric(df_res["grid"], errors="coerce").fillna(0).astype(int)
    
    # Eliminamos duplicados
    df_res = df_res.drop_duplicates(subset=["race_id", "driver_id"], keep="first")
    
    # Convertimos NaN a None
    df_res = df_res.where(pd.notnull(df_res), None)
    return df_res

# ──────────────────────────────────────────────
# 3. LOAD
# ──────────────────────────────────────────────

def load_to_supabase(df: pd.DataFrame):
    table_name = "results"
    print(f"🚀 [LOAD] Subiendo a la tabla '{table_name}'...")
    records = df.to_dict("records")
    batch_size = 500
    
    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        try:
            supabase.table(table_name).upsert(batch).execute()
        except Exception as e:
            print(f"  ❌ Error en el lote {i}: {e}")
            
    print("🏁 [LOAD] ¡Carga completada exitosamente!")

# ──────────────────────────────────────────────
# EJECUCIÓN PRINCIPAL
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== INICIANDO ETL DE FÓRMULA 1 ===\n")
    
    df_bruto = extract_jolpi_results()
    
    if not df_bruto.empty:
        df_limpio = transform_data(df_bruto)
        load_to_supabase(df_limpio)
        
    print("\n=== ETL FINALIZADO ===")