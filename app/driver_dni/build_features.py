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
    print("⏳ Descargando datos de Supabase. Esto puede tomar unos segundos...")
    df_elo     = fetch_table("driver_elo")   
    df_results = fetch_table("results")
    
    print(f"✅ driver_elo descargado: {len(df_elo)} filas totales.")
    print(f"✅ results descargado: {len(df_results)} filas totales.")
    
    # F1 tiene históricamente más de 30,000 registros de carreras por piloto.
    if len(df_elo) < 10000:
        print("⚠️ ADVERTENCIA: Tienes muy pocas filas en driver_elo. Asegúrate de haber subido todo el historial a Supabase.")
        
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
    
    def get_pos(p):
        for col in ["positionOrder", "position"]:
            if col in p and pd.notnull(p[col]) and str(p[col]).strip() not in ["", "\\N"]:
                try: return float(p[col])
                except: pass
        return 99.0
        
    for (race_id, year), grupo in df_completo.groupby(["race_id", "year"]):
        if "constructor_id" not in grupo.columns: continue
            
        grupo_valido = grupo.dropna(subset=["constructor_id"])
        for constructor, equipo in grupo_valido.groupby("constructor_id"):
            if len(equipo) < 2: continue 
                
            pilotos = equipo.to_dict("records")
            for i, p_a in enumerate(pilotos):
                for p_b in pilotos[i+1:]:
                    elo_a = p_a.get("elo_race_final")
                    elo_b = p_b.get("elo_race_final")
                    
                    if pd.isna(elo_a) or pd.isna(elo_b): continue
                        
                    delta_a = elo_a - elo_b
                    pos_a = get_pos(p_a)
                    pos_b = get_pos(p_b)
                    
                    gano_a = 1 if pos_a < pos_b else 0
                    gano_b = 1 if pos_b < pos_a else 0

                    deltas.append({"piloto_id": p_a["piloto_id"], "race_id": race_id, "year": year, "elo_delta_companero": delta_a, "gano_duelo_carrera": gano_a})
                    deltas.append({"piloto_id": p_b["piloto_id"], "race_id": race_id, "year": year, "elo_delta_companero": -delta_a, "gano_duelo_carrera": gano_b})
                    
    if not deltas:
        return pd.DataFrame(columns=["piloto_id", "race_id", "year", "elo_delta_companero", "gano_duelo_carrera"])
    return pd.DataFrame(deltas)

def build_feature_vector(df_elo: pd.DataFrame, df_results: pd.DataFrame) -> pd.DataFrame:
    print("⚙️ Construyendo vectores de features...")

    cols_necesarias = ["race_id", "driver_id", "constructor_id"]
    if "positionOrder" in df_results.columns: cols_necesarias.append("positionOrder")
    elif "position" in df_results.columns: cols_necesarias.append("position")

    res_clean = df_results[cols_necesarias].copy()
    res_clean = res_clean.rename(columns={"driver_id": "piloto_id"})
    
    # 🚨 FIX CRÍTICO: Limpieza indestructible de IDs para que el merge cruce perfectamente
    for df_temp in [df_elo, res_clean]:
        if "race_id" in df_temp.columns:
            # Convierte 10.0 o "10.0" a "10"
            df_temp["race_id"] = df_temp["race_id"].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
        if "piloto_id" in df_temp.columns:
            # Convierte a minúsculas y quita espacios
            df_temp["piloto_id"] = df_temp["piloto_id"].astype(str).str.strip().str.lower()

    res_clean = res_clean.drop_duplicates(subset=["race_id", "piloto_id"], keep="first")
    df = pd.merge(df_elo, res_clean, on=["race_id", "piloto_id"], how="left")
    
    df_deltas = calcular_delta_companero(df)

    records = []
    
    # 🚨 FIX CRÍTICO: Aseguramos numéricos reales para que dropna() y ordenamientos no fallen
    df["elo_race_final"] = pd.to_numeric(df["elo_race_final"], errors="coerce")
    df["race_id_num"] = pd.to_numeric(df["race_id"], errors="coerce").fillna(0)

    for (piloto_id, year), g in df.groupby(["piloto_id", "year"]):
        # Ordenamos numéricamente (1, 2, 3... 10) y no alfabéticamente (1, 10, 2)
        g = g.sort_values("race_id_num")
        elo_finals = g["elo_race_final"].dropna()
        
        if len(elo_finals) < 2:
            continue 

        carreras_totales = len(elo_finals)
        elo_pico         = elo_finals.max()
        elo_final        = elo_finals.iloc[-1]
        elo_volatilidad  = elo_finals.std() if carreras_totales > 1 else 0.0
        elo_trend        = calcular_elo_trend(elo_finals)
        
        if "elo_qualy_final" in g.columns:
            qualy_delta = (g["elo_race_final"] - pd.to_numeric(g["elo_qualy_final"], errors="coerce")).mean()
        else:
            qualy_delta = 0.0

        deltas_piloto = df_deltas[(df_deltas["piloto_id"] == piloto_id) & (df_deltas["year"] == year)]
        elo_delta_comp  = deltas_piloto["elo_delta_companero"].mean() if len(deltas_piloto) else 0.0
        win_rate_carrera = deltas_piloto["gano_duelo_carrera"].mean() if len(deltas_piloto) else 0.5

        records.append({
            "piloto_id"              : str(piloto_id),
            "year"                   : int(year),
            "carreras"               : int(carreras_totales),
            "elo_pico_temporada"     : float(elo_pico),
            "elo_final_temporada"    : float(elo_final),
            "elo_volatilidad"        : float(elo_volatilidad) if pd.notnull(elo_volatilidad) else 0.0,
            "elo_qualy_vs_race_delta": float(qualy_delta) if pd.notnull(qualy_delta) else 0.0,
            "elo_delta_companero"    : float(elo_delta_comp) if pd.notnull(elo_delta_comp) else 0.0,
            "win_rate_duelos_carrera": float(win_rate_carrera) if pd.notnull(win_rate_carrera) else 0.5,
            "elo_trend"              : float(elo_trend),
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
    df_dual = df_dual.fillna(0)
    
    cols_norm = [f"{c}_norm" for c in FEATURE_COLS]
    df_dual[cols_norm] = scaler.fit_transform(df_dual[FEATURE_COLS])
    
    return df_dual, scaler

# ──────────────────────────────────────────────
# 4. EXPORT A SUPABASE Y CSV
# ──────────────────────────────────────────────

def exportar_features(df_dual: pd.DataFrame, scaler: MinMaxScaler):
    df_dual.to_csv("driver_features.csv", index=False)
    joblib.dump(scaler, "scaler.pkl")

    columnas_base = ["piloto_id", "year", "carreras"] + FEATURE_COLS
    df_supabase = df_dual[columnas_base]
    
    records = df_supabase.to_dict("records")
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        supabase.table("driver_features").upsert(batch).execute()

    print(f"\n✓ {len(df_dual)} vectores exportados a Supabase (solo columnas base)")
    print("✓ CSV local y scaler.pkl guardados con las columnas duales para el ML")

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df_elo, df_results = load_raw_data()
    df_features = build_feature_vector(df_elo, df_results)
    
    if not df_features.empty:
        df_dual, scaler = normalizar_features(df_features)

        cols_norm = [f"{c}_norm" for c in FEATURE_COLS]
        print("\nResumen de features (valores normalizados 0-1):")
        print(df_dual[cols_norm].describe().round(3).to_string())

        exportar_features(df_dual, scaler)
    else:
        print("\n⚠️ No se construyó ningún vector.")