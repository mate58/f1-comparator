"""
clustering.py
=============
Pipeline de clustering dinámico para F1 Driver DNA 
(Incluye herramientas de graficación para el Frontend).
"""

import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────
# CONFIGURACIÓN DE DIMENSIONES Y COLORES
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

ARCHETYPE_COLORS = {
    "El Dominador Absoluto": "#E10600",
    "El Maestro a Una Vuelta": "#f39c12",
    "El Muro Defensivo": "#27ae60",
    "El Piloto Escudero": "#2980b9",
    "El Sobreviviente": "#7f8c8d",
    "Desconocido": "#bdc3c7"
}

# ──────────────────────────────────────────────
# VISUALIZACIÓN (Utilizada por Streamlit Frontend)
# ──────────────────────────────────────────────

def graficar_scatter_umap(df: pd.DataFrame, highlight_driver: str = None) -> go.Figure:
    """Genera el gráfico base de UMAP que consumen las páginas del dashboard."""
    # Intentamos asegurar que la columna exista con el nombre correcto
    col_arq = "archetype" if "archetype" in df.columns else "arquetipo"
    
    # Validamos que las coordenadas UMAP existan antes de graficar
    if 'umap_x' not in df.columns or 'umap_y' not in df.columns:
        # Gráfico vacío de advertencia si aún no se corrió dimensionality.py
        fig = go.Figure()
        fig.add_annotation(text="Faltan coordenadas UMAP", showarrow=False, font=dict(size=20))
        return fig

    fig = px.scatter(
        df, 
        x="umap_x", 
        y="umap_y", 
        color=col_arq,
        hover_data=["piloto_id", "year"],
        color_discrete_map=ARCHETYPE_COLORS,
        opacity=0.5
    )
    
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
# LÓGICA DINÁMICA DE ARQUETIPOS
# ──────────────────────────────────────────────

def asignar_nombres_dinamicos(df_centroides: pd.DataFrame) -> dict:
    mapping = {}
    disponibles = list(df_centroides.index)

    c_dom = df_centroides.loc[disponibles, 'elo_pico_temporada'].idxmax()
    mapping[c_dom] = "El Dominador Absoluto"
    disponibles.remove(c_dom)

    c_sob = df_centroides.loc[disponibles, 'elo_pico_temporada'].idxmin()
    mapping[c_sob] = "El Sobreviviente"
    disponibles.remove(c_sob)

    c_esc = df_centroides.loc[disponibles, 'elo_delta_companero'].idxmin()
    mapping[c_esc] = "El Piloto Escudero"
    disponibles.remove(c_esc)

    c_maestro = df_centroides.loc[disponibles, 'elo_qualy_vs_race_delta'].idxmax()
    mapping[c_maestro] = "El Maestro a Una Vuelta"
    disponibles.remove(c_maestro)

    c_muro = disponibles[0]
    mapping[c_muro] = "El Muro Defensivo"

    return mapping

# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    HERE   = Path(__file__).parent
    PARENT = HERE.parent
    
    base_path = PARENT / "driver_features.csv"
    if not base_path.exists():
        raise FileNotFoundError(f"No se encontró {base_path}")
        
    df = pd.read_csv(str(base_path))

    print("Aplicando Feature Weighting a variables de estilo...")
    df_cluster = df[COLS_NORM].copy()
    df_cluster['elo_volatilidad_norm'] = df_cluster['elo_volatilidad_norm'] * 2.0
    df_cluster['elo_qualy_vs_race_delta_norm'] = df_cluster['elo_qualy_vs_race_delta_norm'] * 2.5
    df_cluster['elo_delta_companero_norm'] = df_cluster['elo_delta_companero_norm'] * 1.5

    X_cluster = df_cluster.values

    print("\nEntrenando K-Means forzado con k=5 arquetipos...")
    km = KMeans(n_clusters=5, random_state=42, n_init=20).fit(X_cluster)
    joblib.dump(km, ARTIFACTS_DIR / "kmeans.pkl")

    df["cluster"] = km.predict(X_cluster)
    
    scaler = joblib.load(PARENT / "scaler.pkl")
    centroides_norm = df.groupby("cluster")[COLS_NORM].mean().values
    df_centroides = pd.DataFrame(scaler.inverse_transform(centroides_norm), columns=FEATURE_COLS)
    df_centroides.index = df.groupby("cluster").groups.keys()
    
    mapeo_dinamico = asignar_nombres_dinamicos(df_centroides)
    
   
    df["archetype"] = df["cluster"].map(mapeo_dinamico)
    df["arquetipo"] = df["archetype"]
    
    print("\n=== MAPEO DINÁMICO COMPLETADO ===")
    for k, v in mapeo_dinamico.items():
        print(f"Cluster {k} -> {v}")

    output_path = PARENT / "driver_features_clustered.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ {output_path} guardado con éxito.")