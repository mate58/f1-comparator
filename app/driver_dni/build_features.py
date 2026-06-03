"""
build_features.py
=================
Extrae el vector de features multidimensional por piloto/temporada
a partir de las tablas de Supabase (ELO histórico, resultados, constructores).
"""

import os
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ──────────────────────────────────────────────
# 1. EXTRACCIÓN DESDE SUPABASE
# ──────────────────────────────────────────────

def fetch_table(table_name: str, batch_size: int = 1000) -> pd.DataFrame:
    all_data, start = [], 0
    while True:
        resp = supabase.table(table_name).select("*").range(start, start + batch_size - 1).execute()
        if not resp.data:
            break
        all_data.extend(resp.data)
        if len(resp.data) < batch_size:
            break
        start += batch_size
    return pd.DataFrame(all_data)

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("Cargando datos de Supabase...")
    df_elo     = fetch_table("driver_elo")   
    df_results = fetch_table("results")
    return df_elo, df_results

# ──────────────────────────────────────────────
# 2. CONSTRUCCIÓN DEL VECTOR DE FEATURES
# ──────────────────────────────────────────────

def calcular_elo_trend(elo_series: pd.Series) -> float:
    if len(elo_series) < 2:
        return 0.0
    x = np.arange(len(elo_series))
    slope = np.polyfit(x, elo_series.values, 1)[0]
    return round(float(slope), 4)

def calcular_delta_companero(df_completo: pd.DataFrame) -> pd.DataFrame:
    deltas = []
    
    for (race_id, year), grupo in df_completo.groupby(["race_id", "year"]):
        if "constructor_id" not in grupo.columns:
            continue
            
        for constructor, equipo in grupo.groupby("constructor_id"):
            if len(equipo) < 2:
                continue 
                
            pilotos = equipo.to_dict("records")
            for i, p_a in enumerate(pilotos):
                for p_b in pilotos[i+1:]:
                    # Delta ELO: Quién tiene mejor reputación/ranking acumulado hasta este punto
                    delta_a = p_a["elo_race_final"] - p_b["elo_race_final"]
                    
                    # Duelo de carrera: Quién llegó adelante físicamente en ESTA carrera
                    # Intentamos usar positionOrder (Ergast) o position. Si es nulo, asumimos DNF (99)
                    pos_a = float(p_a.get("positionOrder", p_a.get("position", 99)))
                    pos_b = float(p_b.get("positionOrder", p_b.get("position", 99)))
                    
                    # Si ambos empatan (ej. ambos abandonan), no suma victoria para ninguno (o puedes darle 0.5)
                    gano_a = 1 if pos_a < pos_b else 0
                    gano_b = 1 if pos_b < pos_a else 0

                    deltas.append({
                        "piloto_id": p_a["piloto_id"],
                        "race_id": race_id,
                        "year": year,
                        "elo_delta_companero": delta_a,
                        "gano_duelo_carrera": gano_a
                    })
                    deltas.append({
                        "piloto_id": p_b["piloto_id"],
                        "race_id": race_id,
                        "year": year,
                        "elo_delta_companero": -delta_a,
                        "gano_duelo_carrera": gano_b
                    })
                    
    if not deltas:
        return pd.DataFrame(columns=["piloto_id", "race_id", "year", "elo_delta_companero", "gano_duelo_carrera"])
        
    return pd.DataFrame(deltas)

def build_feature_vector(df_elo: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    print("Construyendo vectores de features...")

    # Extraemos también positionOrder o position para medir los duelos físicos
    cols_necesarias = ["race_id", "driver_id", "constructor_id", "grid"]
    if "positionOrder" in df_results.columns:
        cols_necesarias.append("positionOrder")
    elif "position" in df_results.columns:
        cols_necesarias.append("position")

    res_clean = df_results[cols_necesarias].rename(columns={"driver_id": "piloto_id"})
    df = pd.merge(df_elo, res_clean, on=["race_id", "piloto_id"], how="left")
    df_deltas = calcular_delta_companero(df)

    records = []
    for (piloto_id, year), g in df.groupby(["piloto_id", "year"]):
        g = g.sort_values("race_id")
        elo_finals = g["elo_race_final"].dropna()
        if len(elo_finals) < 3:
            continue 

        elo_pico          = elo_finals.max()
        elo_final         = elo_finals.iloc[-1]
        elo_volatilidad   = elo_finals.std()
        elo_trend         = calcular_elo_trend(elo_finals)
        qualy_delta       = (g["elo_race_final"] - g["elo_qualy_final"]).mean()

        deltas_piloto = df_deltas[(df_deltas["piloto_id"] == piloto_id) & (df_deltas["year"] == year)]
        elo_delta_comp  = deltas_piloto["elo_delta_companero"].mean() if len(deltas_piloto) else 0
        win_rate_carrera = deltas_piloto["gano_duelo_carrera"].mean() if len(deltas_piloto) else 0.5

        records.append({
            "piloto_id"              : piloto_id,
            "year"                   : year,
            "carreras"               : len(elo_finals),
            "elo_pico_temporada"     : elo_pico,
            "elo_final_temporada"    : elo_final,
            "elo_volatilidad"        : elo_volatilidad,
            "elo_qualy_vs_race_delta": qualy_delta,
            "elo_delta_companero"    : elo_delta_comp,
            "win_rate_duelos_carrera": win_rate_carrera,
            "elo_trend"              : elo_trend,
        })

    df_features = pd.DataFrame(records)
    print(f"  → {len(df_features)} vectores construidos ({df_features['piloto_id'].nunique()} pilotos únicos)")
    return df_features

# ──────────────────────────────────────────────
# 3. NORMALIZACIÓN (Patrón Dual)
# ──────────────────────────────────────────────

FEATURE_COLS = [
    "elo_pico_temporada",
    "elo_final_temporada",
    "elo_volatilidad",
    "elo_qualy_vs_race_delta",
    "elo_delta_companero",
    "win_rate_duelos_carrera",
    "elo_trend",
]

def normalizar_features(df_features: pd.DataFrame) -> tuple[pd.DataFrame, MinMaxScaler]:
    scaler = MinMaxScaler()
    df_dual = df_features.copy()
    
    # Creamos sufijos _norm para que los modelos consuman los datos entre [0-1]
    # y el frontend de Streamlit pueda seguir usando los datos reales.
    cols_norm = [f"{c}_norm" for c in FEATURE_COLS]
    df_dual[cols_norm] = scaler.fit_transform(df_features[FEATURE_COLS].fillna(0))
    
    return df_dual, scaler

# ──────────────────────────────────────────────
# 4. EXPORT A SUPABASE Y CSV
# ──────────────────────────────────────────────

def exportar_features(df_dual: pd.DataFrame, scaler: MinMaxScaler):
    df_dual.to_csv("driver_features.csv", index=False)
    joblib.dump(scaler, "scaler.pkl")

    records = df_dual.to_dict("records")
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("driver_features").upsert(batch).execute()

    print(f"\n✓ {len(df_dual)} vectores exportados a Supabase")
    print("✓ scaler.pkl guardado para uso en el dashboard")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df_elo, df_results = load_raw_data()
    df_features = build_feature_vector(df_elo, df_results)
    
    if not df_features.empty:
        df_dual, scaler = normalizar_features(df_features)

        # Mostramos los valores normalizados
        cols_norm = [f"{c}_norm" for c in FEATURE_COLS]
        print("\nResumen de features (valores normalizados 0-1):")
        print(df_dual[cols_norm].describe().round(3).to_string())

        exportar_features(df_dual, scaler)
    else:
        print("\n⚠️ No se construyó ningún vector.")