"""
pages/3_piloto_custom.py
========================
Página 3: Piloto Hipotético
El usuario define un perfil de 7D con sliders y el modelo KNN 
encuentra los pilotos históricos más similares.
"""

import sys
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE    = Path(__file__).resolve().parent
_DNI     = _HERE.parent
_MODELS  = _DNI / "models"

sys.path.insert(0, str(_MODELS))
sys.path.insert(0, str(_DNI))

MODELS_DIR = _DNI
ARTIFACTS_DIR = _MODELS / "artifacts"

from similarity import FEATURE_COLS, FEATURE_LABELS, graficar_barra_similares
from dimensionality import proyectar_nuevo_piloto
from clustering import graficar_scatter_umap

st.set_page_config(page_title="Piloto Hipotético · F1 DNA", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv(MODELS_DIR / "driver_features_clustered.csv")

@st.cache_resource
def cargar_modelos():
    knn    = joblib.load(ARTIFACTS_DIR / "knn.pkl")
    return knn

df  = cargar_datos()
knn = cargar_modelos()
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎛️ Piloto Hipotético")
st.markdown("""
Definí el perfil de un piloto usando los sliders. El sistema proyectará ese perfil
en el espacio UMAP y encontrará los pilotos históricos más parecidos.
""")

# ── Presets ───────────────────────────────────────────────────────────────────
st.markdown("#### Presets rápidos")
preset_cols = st.columns(5)
# Presets adaptados a 7 dimensiones
presets = {
    "Dominador"     : {"elo_pico":0.95, "elo_final":0.93, "volatilidad":0.1, "qualy_race":0.6, "companero":0.85, "win_rate":0.90, "trend":0.5},
    "Sobre-rendidor": {"elo_pico":0.60, "elo_final":0.58, "volatilidad":0.3, "qualy_race":0.7, "companero":0.75, "win_rate":0.78, "trend":0.55},
    "Clasificador"  : {"elo_pico":0.70, "elo_final":0.65, "volatilidad":0.2, "qualy_race":0.2, "companero":0.55, "win_rate":0.60, "trend":0.4},
    "Consistente"   : {"elo_pico":0.55, "elo_final":0.54, "volatilidad":0.05,"qualy_race":0.5, "companero":0.50, "win_rate":0.55, "trend":0.1},
    "Rookie"        : {"elo_pico":0.35, "elo_final":0.40, "volatilidad":0.7, "qualy_race":0.4, "companero":0.30, "win_rate":0.35, "trend":0.90},
}

selected_preset = None
for col, (nombre, vals) in zip(preset_cols, presets.items()):
    with col:
        if st.button(nombre, use_container_width=True):
            selected_preset = vals

# ── Sliders ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Configurá el perfil (7 Dimensiones)")
st.caption("Todos los valores están normalizados en [0, 1]. El modelo los mapea al espacio de features.")

def get_default(field: str, default: float) -> float:
    return selected_preset.get(field, default) if selected_preset else default

c1, c2, c3 = st.columns(3)

with c1:
    elo_pico    = st.slider("ELO pico temporada", 0.0, 1.0, get_default("elo_pico", 0.5), 0.01)
    elo_final   = st.slider("ELO final temporada", 0.0, 1.0, get_default("elo_final", 0.48), 0.01)
    volatilidad = st.slider("Consistencia (inverso volatilidad)", 0.0, 1.0, get_default("volatilidad", 0.5), 0.01)

with c2:
    qualy_race  = st.slider("Mejora qualy → carrera", 0.0, 1.0, get_default("qualy_race", 0.5), 0.01)
    companero   = st.slider("Delta vs. compañero", 0.0, 1.0, get_default("companero", 0.5), 0.01)

with c3:
    win_rate    = st.slider("Win rate duelos", 0.0, 1.0, get_default("win_rate", 0.5), 0.01)
    trend       = st.slider("Progresión en la temporada", 0.0, 1.0, get_default("trend", 0.5), 0.01)

# ── Construir vector 7D ───────────────────────────────────────────────────────
vector_custom = {
    "elo_pico_temporada": elo_pico,
    "elo_final_temporada": elo_final,
    "elo_volatilidad": 1.0 - volatilidad,  # Invertido para que el KNN entienda
    "elo_qualy_vs_race_delta": qualy_race,
    "elo_delta_companero": companero,
    "win_rate_duelos_carrera": win_rate,
    "elo_trend": trend,
}

st.divider()

# ── Buscar matches ────────────────────────────────────────────────────────────
if st.button("🔍 Encontrar pilotos similares", type="primary", use_container_width=True):

    # 1. Ejecutamos el KNN manualmente
    vector_array = np.array([[vector_custom[col] for col in FEATURE_COLS]])
    distancias, indices = knn.kneighbors(vector_array, n_neighbors=5)
    
    resultados = []
    for dist, idx in zip(distancias[0], indices[0]):
        fila_c = df.iloc[idx]
        similitud = max(0.0, (1.0 - dist) * 100.0)
        resultados.append({
            "piloto_id": fila_c["piloto_id"],
            "year": fila_c["year"],
            "similitud_pct": round(similitud, 1),
            "archetype": fila_c.get("archetype", "—")
        })
    df_matches = pd.DataFrame(resultados)

    col_res, col_radar = st.columns([1, 1])

    with col_res:
        st.subheader("🏆 Pilotos históricos más similares")
        fig_barras = graficar_barra_similares(df_matches, titulo="Match de pilotos históricos")
        st.plotly_chart(fig_barras, use_container_width=True)

        for _, row in df_matches.iterrows():
            st.markdown(f"""
            <div style="padding:8px 14px; margin:4px 0; border-radius:8px; background:#f0f2f6;
                        border-left:3px solid #E10600;">
                <b>{row['piloto_id']}</b> ({row['year']}) ·
                <span style="color:#E10600; font-weight:600">{row['similitud_pct']}%</span> ·
                {row['archetype']}
            </div>""", unsafe_allow_html=True)

    with col_radar:
        st.subheader("🧬 Tu perfil vs. mejor match")
        mejor = df_matches.iloc[0]
        
        # Obtenemos vector de match desde columnas normalizadas
        fila_match = df[(df["piloto_id"].astype(str) == str(mejor["piloto_id"])) & (df["year"] == int(mejor["year"]))].iloc[0]
        
        vec_match_radar = []
        vec_custom_radar = []
        labels = []
        
        for f in FEATURE_COLS:
            val_c = vector_custom[f]
            val_m = float(fila_match[f"{f}_norm"])
            if f == "elo_volatilidad":
                val_c = 1.0 - val_c
                val_m = 1.0 - val_m
            vec_custom_radar.append(val_c)
            vec_match_radar.append(val_m)
            labels.append(FEATURE_LABELS.get(f, f))

        cats = labels + [labels[0]]
        vc = vec_custom_radar + [vec_custom_radar[0]]
        vm = vec_match_radar  + [vec_match_radar[0]]

        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(
            r=vc, theta=cats, fill="toself", name="Tu piloto",
            fillcolor="rgba(225,6,0,0.15)", line=dict(color="#E10600", width=2)
        ))
        fig_r.add_trace(go.Scatterpolar(
            r=vm, theta=cats, fill="toself", name=f"{mejor['piloto_id']} ({mejor['year']})",
            fillcolor="rgba(0,144,208,0.15)", line=dict(color="#0090D0", width=2)
        ))
        fig_r.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=True, template="plotly_white", height=400,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_r, use_container_width=True)

    # ── Scatter UMAP con piloto hipotético proyectado ──────────────────────
    st.divider()
    st.subheader("📍 Tu piloto en el espacio UMAP")
    st.caption("La estrella ★ muestra dónde caería tu perfil en el mapa de arquetipos históricos.")

    try:
        coords_2d = proyectar_nuevo_piloto(vector_array, str(ARTIFACTS_DIR / "umap_2d.pkl"))

        fig_sc = graficar_scatter_umap(df)
        fig_sc.add_trace(go.Scatter(
            x=[coords_2d[0, 0]], y=[coords_2d[0, 1]], mode="markers+text",
            name="★ Tu piloto", marker=dict(size=18, color="#FFD700", symbol="star", line=dict(width=2, color="#E10600")),
            text=["Tu perfil"], textposition="top center"
        ))
        st.plotly_chart(fig_sc, use_container_width=True)
    except Exception as e:
        st.info(f"No se pudo proyectar el piloto en el mapa UMAP. Detalle: {e}")