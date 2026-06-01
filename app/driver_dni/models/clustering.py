"""
clustering.py
=============
Pipeline de clustering para el proyecto F1 Driver DNA.
"""

import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Resuelve artifacts/ relativo a este archivo, no al cwd.
# Así funciona igual si se llama desde app/, desde models/, o desde la raíz.
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

ARCHETYPE_NAMES = {
    0: "Dominador absoluto",
    1: "Sobre-rendidor con mal auto",
    2: "Clasificador puro",
    3: "Consistente de midfield",
    4: "Piloto de desarrollo",
    5: "Piloto irregular",
}

FEATURE_COLS = [
    "elo_pico_temporada",
    "elo_final_temporada",
    "elo_volatilidad",
    "elo_qualy_vs_race_delta",
    "elo_delta_companero",
    "win_rate_duelos_carrera",
    "car_bonus_avg",
    "elo_trend",
    "remontada_rate",
]



# ──────────────────────────────────────────────
# BÚSQUEDA DEL K ÓPTIMO
# ──────────────────────────────────────────────

def buscar_k_optimo(
    X: np.ndarray,
    k_min: int = 2,
    k_max: int = 10,
    random_state: int = 42
) -> tuple[int, pd.DataFrame]:
    print(f"Buscando k óptimo en rango [{k_min}, {k_max}]...")
    resultados = []

    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        sil    = silhouette_score(X, labels)
        inercia = km.inertia_

        resultados.append({"k": k, "silhouette": sil, "inercia": inercia})
        print(f"  k={k} | Silhouette={sil:.4f} | Inercia={inercia:.1f}")

    df_metricas = pd.DataFrame(resultados)
    k_optimo    = int(df_metricas.loc[df_metricas["silhouette"].idxmax(), "k"])

    print(f"\n✓ k óptimo: {k_optimo} (Silhouette={df_metricas.loc[df_metricas.k==k_optimo, 'silhouette'].values[0]:.4f})")
    return k_optimo, df_metricas

# ──────────────────────────────────────────────
# ENTRENAMIENTO DEL MODELO FINAL
# ──────────────────────────────────────────────

def entrenar_kmeans(X: np.ndarray, k: int, random_state: int = 42) -> KMeans:
    print(f"\nEntrenando K-Means con k={k}...")
    km = KMeans(n_clusters=k, random_state=random_state, n_init=20)
    km.fit(X)

    joblib.dump(km, ARTIFACTS_DIR / "kmeans.pkl")
    print(f"  → Inercia final: {km.inertia_:.2f}")
    print(f"  → Silhouette Score: {silhouette_score(X, km.labels_):.4f}")
    print("  → Modelo guardado en artifacts/kmeans.pkl")
    return km

# ──────────────────────────────────────────────
# ANÁLISIS DE CENTROIDES Y ASIGNACIÓN
# ──────────────────────────────────────────────

def asignar_arquetipos(df_proyectado: pd.DataFrame, km: KMeans, X_cluster: np.ndarray) -> pd.DataFrame:
    """Asigna los clústeres calculados de vuelta al DataFrame original."""
    df_proyectado = df_proyectado.copy()
    df_proyectado["cluster"]   = km.predict(X_cluster)
    df_proyectado["archetype"] = df_proyectado["cluster"].map(
        lambda i: ARCHETYPE_NAMES.get(i, f"Cluster {i}")
    )
    return df_proyectado


ARCHETYPE_COLORS = {
    "Dominador absoluto"           : "#E10600",
    "Sobre-rendidor con mal auto"  : "#FF8700",
    "Clasificador puro"            : "#0090D0",
    "Consistente de midfield"      : "#00D2BE",
    "Piloto de desarrollo"         : "#9B0A4B",
    "Piloto irregular"             : "#888888",
}


def graficar_scatter_umap(df: pd.DataFrame, highlight_driver: str | None = None) -> go.Figure:
    """
    Scatter plot 2D del espacio UMAP coloreado por arquetipo.
    Hover muestra piloto_id, year, ELO y archetype.
    Si se provee highlight_driver, ese piloto se resalta con una estrella.
    """
    fig = go.Figure()

    archetype_col = "archetype" if "archetype" in df.columns else None

    grupos = df.groupby("archetype") if archetype_col else [("Sin cluster", df)]
    for archetype, grupo in grupos:
        color = ARCHETYPE_COLORS.get(archetype, "#AAAAAA")
        fig.add_trace(go.Scatter(
            x=grupo["umap_x"],
            y=grupo["umap_y"],
            mode="markers",
            name=str(archetype),
            marker=dict(size=7, color=color, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            customdata=grupo[["piloto_id", "year", "elo_final_temporada"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "ELO final: %{customdata[2]:.0f}<br>"
                f"Arquetipo: {archetype}<extra></extra>"
            )
        ))

    if highlight_driver is not None:
        piloto_df = df[df["piloto_id"].astype(str) == str(highlight_driver)]
        if not piloto_df.empty:
            fig.add_trace(go.Scatter(
                x=piloto_df["umap_x"],
                y=piloto_df["umap_y"],
                mode="markers+text",
                name=f"▶ {highlight_driver}",
                marker=dict(size=14, color="#E10600", symbol="star",
                            line=dict(width=2, color="white")),
                text=piloto_df["year"].astype(str),
                textposition="top center",
                showlegend=True
            ))

    fig.update_layout(
        title="Espacio UMAP de pilotos F1 (2000–2024)",
        xaxis_title="UMAP Dimensión 1",
        yaxis_title="UMAP Dimensión 2",
        legend_title="Arquetipo",
        template="plotly_white",
        height=550,
    )
    return fig


def analizar_centroides(df_clustered: pd.DataFrame, scaler) -> pd.DataFrame:
    """
    Calcula el centroide real en 9D agrupando los vectores normalizados por cluster
    y luego usando el scaler para llevarlos a valores interpretables.
    """
    # 1. Agrupamos las 9 features originales por cluster y sacamos el promedio
    centroides_norm_9d = df_clustered.groupby("cluster")[FEATURE_COLS].mean().values
    
    # 2. Desnormalizamos esos promedios de 9D al espacio real
    centroides_real = scaler.inverse_transform(centroides_norm_9d)

    df_centroides = pd.DataFrame(centroides_real, columns=FEATURE_COLS)
    
    # Aseguramos el orden de los clústeres
    df_centroides.index = df_clustered.groupby("cluster").groups.keys()
    df_centroides.index.name = "cluster"
    df_centroides["archetype"] = df_centroides.index.map(
        lambda i: ARCHETYPE_NAMES.get(i, f"Cluster {i}")
    )
    return df_centroides



# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    # Detección inteligente de archivo (por si no hiciste UMAP aún)
    if Path("driver_features_umap.csv").exists():
        df = pd.read_csv("driver_features_umap.csv")
    else:
        print("⚠️ No se detectó archivo UMAP. Leyendo 'driver_features.csv'...")
        df = pd.read_csv("driver_features.csv")
        
    scaler = joblib.load("scaler.pkl")

    # Adaptación automática: usa 3D si existe UMAP, sino usa 9D
    if "umap_x3" in df.columns:
        print("Clustering sobre espacio UMAP 3D...")
        X_cluster = df[["umap_x3", "umap_y3", "umap_z3"]].values
    else:
        print("Clustering sobre las 9 dimensiones originales...")
        X_cluster = df[FEATURE_COLS].values

    # 1. Buscar K
    k_optimo, df_metricas = buscar_k_optimo(X_cluster)
    
    # 2. Entrenar modelo
    km = entrenar_kmeans(X_cluster, k_optimo)

    # 3. Asignar arquetipos PRIMERO (para tener la columna cluster)
    df_final = asignar_arquetipos(df, km, X_cluster)
    
    # 4. Analizar centroides DESPUÉS (requiere la columna cluster en df_final)
    df_centroides = analizar_centroides(df_final, scaler)
    
    print("\n=== CENTROIDES (valores reales desnormalizados) ===")
    print(df_centroides.round(2).to_string())

    df_final.to_csv("driver_features_clustered.csv", index=False)
    print(f"\n✓ driver_features_clustered.csv guardado con {df_final['cluster'].nunique()} clusters")