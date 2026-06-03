"""
clustering.py
=============
Pipeline de clustering para el proyecto F1 Driver DNA.
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARCHETYPE_NAMES = {
    0: "Dominador absoluto",
    1: "Sobre-rendidor con mal auto",
    2: "Clasificador puro",
    3: "Consistente de midfield",
    4: "Piloto de desarrollo",
    5: "Piloto irregular",
}

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE DIMENSIONES (7D)
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
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]


# ──────────────────────────────────────────────
# BÚSQUEDA DEL K ÓPTIMO Y ENTRENAMIENTO
# ──────────────────────────────────────────────

def buscar_k_optimo(X: np.ndarray, k_min: int = 2, k_max: int = 10, random_state: int = 42) -> tuple[int, pd.DataFrame]:
    print(f"Buscando k óptimo en rango [{k_min}, {k_max}]...")
    resultados = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        resultados.append({"k": k, "silhouette": sil, "inercia": km.inertia_})
        print(f"  k={k} | Silhouette={sil:.4f} | Inercia={km.inertia_:.1f}")

    df_metricas = pd.DataFrame(resultados)
    k_optimo = int(df_metricas.loc[df_metricas["silhouette"].idxmax(), "k"])
    print(f"\n✓ k óptimo: {k_optimo}")
    return k_optimo, df_metricas


def entrenar_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    print(f"\nEntrenando K-Means con k={k}...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20).fit(X)
    joblib.dump(km, ARTIFACTS_DIR / "kmeans.pkl")
    return km


def analizar_centroides(df_clustered: pd.DataFrame, scaler) -> pd.DataFrame:
    # 1. Agrupamos usando las features normalizadas por cluster
    centroides_norm = df_clustered.groupby("cluster")[COLS_NORM].mean().values
    
    # 2. Desnormalizamos para que sea legible por humanos
    centroides_real = scaler.inverse_transform(centroides_norm)
    df_centroides = pd.DataFrame(centroides_real, columns=FEATURE_COLS)
    
    df_centroides.index = df_clustered.groupby("cluster").groups.keys()
    df_centroides.index.name = "cluster"
    df_centroides["archetype"] = df_centroides.index.map(lambda i: ARCHETYPE_NAMES.get(i, f"Cluster {i}"))
    return df_centroides


# ──────────────────────────────────────────────
# COLORES Y VISUALIZACIÓN (PARA STREAMLIT)
# ──────────────────────────────────────────────

ARCHETYPE_COLORS = {
    "Dominador absoluto": "#E10600",        # Rojo F1
    "Sobre-rendidor con mal auto": "#2980b9", # Azul
    "Clasificador puro": "#f39c12",         # Naranja
    "Consistente de midfield": "#27ae60",     # Verde
    "Piloto de desarrollo": "#8e44ad",        # Morado
    "Piloto irregular": "#7f8c8d",            # Gris
    "Desconocido": "#bdc3c7"
}

def graficar_scatter_umap(df: pd.DataFrame, highlight_driver: str = None) -> go.Figure:
    """
    Crea un Scatter Plot 2D de las coordenadas UMAP coloreado por arquetipo.
    Si se pasa un highlight_driver, resalta sus temporadas con estrellas.
    """
    fig = px.scatter(
        df, 
        x="umap_x", 
        y="umap_y", 
        color="archetype",
        hover_data=["piloto_id", "year"],
        color_discrete_map=ARCHETYPE_COLORS,
        opacity=0.5
    )
    
    # Resaltar al piloto seleccionado (estrella dorada)
    if highlight_driver:
        df_hl = df[df["piloto_id"].astype(str) == str(highlight_driver)]
        if not df_hl.empty:
            fig.add_trace(go.Scatter(
                x=df_hl["umap_x"],
                y=df_hl["umap_y"],
                mode="markers",
                marker=dict(
                    size=14, 
                    color="gold", 
                    line=dict(width=2, color="black"), 
                    symbol="star"
                ),
                name=f"★ {highlight_driver}",
                text=df_hl["year"],
                hovertemplate="<b>%{text}</b><extra></extra>"
            ))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Dimensión UMAP 1",
        yaxis_title="Dimensión UMAP 2",
        legend_title="Arquetipo"
    )
    
    return fig


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def leer_csv_seguro(path: str) -> pd.DataFrame:
    # Lectura genérica delegada al header del CSV
    return pd.read_csv(path)

if __name__ == "__main__":
    HERE   = Path(__file__).parent
    PARENT = HERE.parent
    
    # Leemos el archivo con los datos de UMAP ya generados
    base_path = PARENT / "driver_features_umap.csv"
    if not base_path.exists():
        base_path = PARENT / "driver_features.csv"
        
    if not base_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en {base_path}")
        
    df = leer_csv_seguro(str(base_path))

    # Aseguramos que tenemos las columnas _norm
    if not all(col in df.columns for col in COLS_NORM):
        raise ValueError("Faltan las columnas normalizadas (_norm). Ejecuta build_features.py primero.")

    print("Clustering sobre las 7 dimensiones reales (normalizadas)...")
    X_cluster = df[COLS_NORM].values

    # K-Means
    k_optimo, _ = buscar_k_optimo(X_cluster)
    km = entrenar_kmeans(X_cluster, k_optimo)

    # Asignar a nuestro DataFrame
    df["cluster"] = km.predict(X_cluster)
    df["archetype"] = df["cluster"].map(lambda i: ARCHETYPE_NAMES.get(i, f"Cluster {i}"))
    
    # Desnormalizar centroides para la consola
    scaler = joblib.load(PARENT / "scaler.pkl")
    df_centroides = analizar_centroides(df, scaler)
    
    print("\n=== CENTROIDES (valores reales desnormalizados) ===")
    print(df_centroides.round(2).to_string())

    output_path = PARENT / "driver_features_clustered.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ {output_path} guardado con éxito.")