"""
etl_metadata.py
===============
Descarga las tablas maestras (Pilotos, Escuderías, Carreras) 
desde Jolpi API y las sube a Supabase.
"""

import os
import time
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

# ──────────────────────────────────────────────
# FUNCIONES DE EXTRACCIÓN (JOLPI API)
# ──────────────────────────────────────────────

def fetch_jolpi_endpoint(endpoint: str, data_key: str, list_key: str) -> list:
    """Función genérica para paginar endpoints de Jolpi API."""
    print(f"⏳ Descargando /api/f1/{endpoint}.json ...")
    base_url = f"http://api.jolpi.ca/ergast/f1/{endpoint}.json"
    

    limit = 100
    offset = 0
    total = 1
    all_items = []
    
    while offset < total:
        response = requests.get(f"{base_url}?limit={limit}&offset={offset}")
        if response.status_code != 200:
            print(f"❌ Error en API: {response.status_code}")
            break
            
        data = response.json()
        mrdata = data.get("MRData", {})
        total = int(mrdata.get("total", 0))
        
        items = mrdata.get(data_key, {}).get(list_key, [])
        if not items:
            break
            
        all_items.extend(items)
        
        print(f"  → Procesados {len(all_items)} de {total}...")
        
        # 🚨 FIX: Avanzamos el offset exactamente la cantidad que nos devolvieron
        offset += limit
        time.sleep(0.5)
        
    return all_items

# ──────────────────────────────────────────────
# PIPELINES ESPECÍFICOS
# ──────────────────────────────────────────────

def etl_drivers():
    raw_data = fetch_jolpi_endpoint("drivers", "DriverTable", "Drivers")
    df = pd.DataFrame(raw_data)
    
    # 🚨 FIX CRÍTICO: Renombrar permanentNumber a number
    if "permanentNumber" in df.columns:
        df = df.rename(columns={"permanentNumber": "number"})
        
    df = df.rename(columns={"driverId": "driver_id"})
    df["driver_id"] = df["driver_id"].astype(str).str.strip().str.lower()
    
    # Filtramos estrictamente las columnas que existen en Supabase
    columnas_sql = ["driver_id", "driverRef", "number", "code", "givenName", "familyName", "dateOfBirth", "nationality", "url"]
    columnas_finales = [c for c in columnas_sql if c in df.columns]
    
    return df[columnas_finales]

def etl_constructors():
    raw_data = fetch_jolpi_endpoint("constructors", "ConstructorTable", "Constructors")
    df = pd.DataFrame(raw_data)
    
    df = df.rename(columns={"constructorId": "constructor_id"})
    df["constructor_id"] = df["constructor_id"].astype(str).str.strip().str.lower()
    
    columnas_sql = ["constructor_id", "constructorRef", "name", "nationality", "url"]
    columnas_finales = [c for c in columnas_sql if c in df.columns]
    
    return df[columnas_finales]

def etl_races():
    raw_data = fetch_jolpi_endpoint("races", "RaceTable", "Races")
    records = []
    
    for race in raw_data:
        records.append({
            "race_id": f"{race.get('season')}_{race.get('round')}",
            "year": int(race.get("season")),
            "round": int(race.get("round")),
            "circuit_id": race.get("Circuit", {}).get("circuitId"),
            "circuitName": race.get("Circuit", {}).get("circuitName"),
            "raceName": race.get("raceName"),
            "date": race.get("date"),
            "time": race.get("time")
        })
        
    return pd.DataFrame(records)

# ──────────────────────────────────────────────
# FUNCIÓN DE CARGA
# ──────────────────────────────────────────────

def load_to_supabase(df: pd.DataFrame, table_name: str):
    print(f"🚀 Subiendo a la tabla '{table_name}'...")
    records = df.to_dict("records")
    
    # Limpiamos NaNs
    cleaned_records = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if pd.isna(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned_records.append(clean_row)
        
    batch_size = 500
    
    for i in range(0, len(cleaned_records), batch_size):
        batch = cleaned_records[i : i + batch_size]
        try:
            supabase.table(table_name).upsert(batch).execute()
        except Exception as e:
            print(f"  ❌ Error subiendo lote {i}: {e}")
            
    print(f"✅ {table_name} completado: {len(cleaned_records)} registros.\n")

# ──────────────────────────────────────────────
# EJECUCIÓN
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=== INICIANDO EXTRACCIÓN DE METADATA ===\n")
    
    df_drivers = etl_drivers()
    load_to_supabase(df_drivers, "drivers")
    
    df_constructors = etl_constructors()
    load_to_supabase(df_constructors, "constructors")
    
    df_races = etl_races()
    load_to_supabase(df_races, "races")
    
    print("=== METADATA LISTA ===")