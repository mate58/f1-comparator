"""
similarity.py
=============
Motor de similitud y búsqueda de pilotos "gemelos" (Nearest Neighbors).
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE DIMENSIONES (7D REALES)
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

# Usamos las columnas normalizadas para los cálculos matemáticos
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]

FEATURE_LABELS = {
    "elo_pico_temporada": "Pico de Nivel",
    "elo_final_temporada": "Nivel Final",
    "elo_volatilidad": "Consistencia", # Nota: Invertir visualmente (1-v) en el radar
    "elo_qualy_vs_race_delta": "Ritmo de Carrera",
    "elo_delta_companero": "Dominio de Equipo",
    "win_rate_duelos_carrera": "Duelos Ganados",
    "elo_trend": "Progresión",
}

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# ENTRENAMIENTO DEL MODELO
# ──────────────────────────────────────────────

def entrenar_knn(df_features: pd.DataFrame, metric="cosine") -> NearestNeighbors:
    """
    Entrena el modelo KNN sobre las dimensiones normalizadas.
    La métrica 'cosine' es ideal para evaluar la "forma" del perfil del piloto.
    """
    X = df_features[COLS_NORM].fillna(0).values
    print(f"Entrenando KNN ({metric}) sobre {X.shape[0]} vectores de {X.shape[1]} dimensiones...")
    
    # algorithm="brute" es seguro y rápido para cosine distance en datasets de F1
    knn = NearestNeighbors(n_neighbors=10, metric=metric, algorithm="brute")
    knn.fit(X)
    
    joblib.dump(knn, ARTIFACTS_DIR / "knn.pkl")
    print("  → Modelo KNN guardado en artifacts/knn.pkl")
    
    return knn


# ──────────────────────────────────────────────
# BÚSQUEDA DE SIMILARES
# ──────────────────────────────────────────────

def top_similares(
    piloto_id: str, 
    year: int, 
    df: pd.DataFrame, 
    knn: NearestNeighbors, 
    n: int = 5
) -> pd.DataFrame:
    """
    Busca los n pilotos más parecidos a un piloto_id/year específico.
    """
    mask = (df["piloto_id"].astype(str) == str(piloto_id)) & (df["year"] == int(year))
    if not mask.any():
        raise ValueError(f"No se encontró el piloto '{piloto_id}' en el año {year}.")
    
    # Extraemos el vector NORMALIZADO del piloto buscado
    vector_query = df.loc[mask, COLS_NORM].fillna(0).values

    # Buscamos n+1 porque el primer resultado siempre es el propio piloto (distancia 0)
    distancias, indices = knn.kneighbors(vector_query, n_neighbors=n + 1)

    resultados = []
    # Iteramos saltando el primero [1:] si es el mismo piloto
    # (a veces puede fallar si hay duplicados exactos, pero manejamos la lógica general)
    for dist, idx in zip(distancias[0], indices[0]):
        fila_candidato = df.iloc[idx]
        
        # Filtramos para no mostrar al mismo piloto en el mismo año
        if str(fila_candidato["piloto_id"]) == str(piloto_id) and int(fila_candidato["year"]) == int(year):
            continue
            
        # Convertimos distancia Coseno a Similitud %
        # Distancia Coseno va de 0 (idénticos) a 2 (opuestos).
        # Porcentaje = (1 - dist) * 100
        similitud = max(0.0, (1.0 - dist) * 100.0)
        
        resultados.append({
            "piloto_id": fila_candidato["piloto_id"],
            "year": fila_candidato["year"],
            "similitud_pct": round(similitud, 1),
            "archetype": fila_candidato.get("archetype", "Desconocido")
        })

    # Si por alguna razón devolvió n+1 distintos (no saltó el original), cortamos a n
    df_res = pd.DataFrame(resultados).head(n)
    return df_res


# ──────────────────────────────────────────────
# GRAFICACIÓN (PARA STREAMLIT)
# ──────────────────────────────────────────────

def graficar_barra_similares(df_similares: pd.DataFrame, titulo: str = "Similitud") -> go.Figure:
    """
    Crea un gráfico de barras horizontal (Plotly) para los pilotos más parecidos.
    """
    df_plot = df_similares.copy()
    # Invertimos para que el más parecido salga arriba
    df_plot = df_plot.iloc[::-1]
    
    # Etiqueta bonita "Schumacher (2001)"
    nombres = df_plot.apply(lambda row: f"{row['piloto_id']} ({row['year']})", axis=1)

    fig = go.Figure(go.Bar(
        x=df_plot["similitud_pct"],
        y=nombres,
        orientation="h",
        marker_color="#2980b9",
        text=df_plot["similitud_pct"].apply(lambda x: f"{x}%"),
        textposition="outside"
    ))
    
    fig.update_layout(
        title=titulo,
        xaxis=dict(range=[0, 105], title="% Similitud"),
        yaxis=dict(title=""),
        template="plotly_white",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    HERE = Path(__file__).parent
    PARENT = HERE.parent
    
    csv_path = PARENT / "driver_features_clustered.csv"
    if not csv_path.exists():
        csv_path = PARENT / "driver_features.csv"
        
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en {csv_path}")

    df = pd.read_csv(csv_path)
    
    # Verificamos que las columnas existan
    if not all(col in df.columns for col in COLS_NORM):
        raise ValueError("El CSV no contiene las columnas '_norm'. Ejecuta build_features.py primero.")

    knn = entrenar_knn(df)
    
    # Test rápido de validación
    print("\n--- Test Rápido ---")
    try:
        # Busca un piloto de prueba que exista, por ejemplo:
        test_driver = df["piloto_id"].iloc[0]
        test_year = df["year"].iloc[0]
        print(f"Buscando parecidos a {test_driver} ({test_year})...")
        res = top_similares(test_driver, test_year, df, knn)
        print(res.to_string(index=False))
    except Exception as e:
        print(f"Error en el test: {e}")