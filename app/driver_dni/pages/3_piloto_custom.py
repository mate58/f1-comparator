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
from clustering import graficar_scatter_umap

st.set_page_config(page_title="Piloto Hipotético · F1 DNA", layout="wide")

@st.cache_data
def cargar_datos():
    ruta = MODELS_DIR / "driver_features_final.csv"
    if not ruta.exists():
        ruta = MODELS_DIR / "driver_features_clustered.csv"
        
    df = pd.read_csv(ruta)
    if "arquetipo" in df.columns and "archetype" not in df.columns:
        df = df.rename(columns={"arquetipo": "archetype"})
    return df

@st.cache_resource
def cargar_modelos():
    knn    = joblib.load(ARTIFACTS_DIR / "knn.pkl")
    return knn

df  = cargar_datos()
knn = cargar_modelos()
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]

# ── FIX: Inicialización de Memoria (Session State) ───────────────────────────
if "sl_pico" not in st.session_state:
    st.session_state.update({
        "sl_pico": 0.5, "sl_final": 0.48, "sl_vol": 0.5,
        "sl_qualy": 0.5, "sl_comp": 0.5, "sl_win": 0.5, "sl_trend": 0.5
    })

# Callback: Cuando apretamos un preset, guarda los valores en la memoria
def aplicar_preset(vals):
    st.session_state.sl_pico = vals["elo_pico"]
    st.session_state.sl_final = vals["elo_final"]
    st.session_state.sl_vol = vals["volatilidad"]
    st.session_state.sl_qualy = vals["qualy_race"]
    st.session_state.sl_comp = vals["companero"]
    st.session_state.sl_win = vals["win_rate"]
    st.session_state.sl_trend = vals["trend"]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎛️ Piloto Hipotético")
st.markdown("""
Definí el perfil de un piloto usando los sliders. El sistema encontrará los pilotos históricos 
más parecidos y te mostrará en qué sector del mapa UMAP caería tu selección.
""")

# ── Presets ───────────────────────────────────────────────────────────────────
st.markdown("#### Presets rápidos")
preset_cols = st.columns(5)
presets = {
    "Dominador"     : {"elo_pico":0.95, "elo_final":0.93, "volatilidad":0.1, "qualy_race":0.6, "companero":0.85, "win_rate":0.90, "trend":0.5},
    "Sobre-rendidor": {"elo_pico":0.60, "elo_final":0.58, "volatilidad":0.3, "qualy_race":0.7, "companero":0.75, "win_rate":0.78, "trend":0.55},
    "Clasificador"  : {"elo_pico":0.70, "elo_final":0.65, "volatilidad":0.2, "qualy_race":0.2, "companero":0.55, "win_rate":0.60, "trend":0.4},
    "Consistente"   : {"elo_pico":0.55, "elo_final":0.54, "volatilidad":0.05,"qualy_race":0.5, "companero":0.50, "win_rate":0.55, "trend":0.1},
    "Rookie"        : {"elo_pico":0.35, "elo_final":0.40, "volatilidad":0.7, "qualy_race":0.4, "companero":0.30, "win_rate":0.35, "trend":0.90},
}

# Modificamos los botones para usar el "on_click" callback
for col, (nombre, vals) in zip(preset_cols, presets.items()):
    with col:
        st.button(nombre, on_click=aplicar_preset, args=(vals,), use_container_width=True)

# ── Sliders ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Configurá el perfil (7 Dimensiones)")
st.caption("Todos los valores están normalizados en [0, 1]. El modelo los mapea al espacio de features.")

c1, c2, c3 = st.columns(3)

# Ahora los sliders usan las variables "key", lo que los conecta directamente a la memoria
with c1:
    elo_pico    = st.slider("ELO pico temporada", 0.0, 1.0, key="sl_pico", step=0.01)
    elo_final   = st.slider("ELO final temporada", 0.0, 1.0, key="sl_final", step=0.01)
    volatilidad = st.slider("Consistencia (inverso volatilidad)", 0.0, 1.0, key="sl_vol", step=0.01)

with c2:
    qualy_race  = st.slider("Mejora qualy → carrera", 0.0, 1.0, key="sl_qualy", step=0.01)
    companero   = st.slider("Delta vs. compañero", 0.0, 1.0, key="sl_comp", step=0.01)

with c3:
    win_rate    = st.slider("Win rate duelos", 0.0, 1.0, key="sl_win", step=0.01)
    trend       = st.slider("Progresión en la temporada", 0.0, 1.0, key="sl_trend", step=0.01)

vector_custom = {
    "elo_pico_temporada": elo_pico,
    "elo_final_temporada": elo_final,
    "elo_volatilidad": 1.0 - volatilidad,
    "elo_qualy_vs_race_delta": qualy_race,
    "elo_delta_companero": companero,
    "win_rate_duelos_carrera": win_rate,
    "elo_trend": trend,
}

st.divider()

# ── Buscar matches ────────────────────────────────────────────────────────────
if st.button("🔍 Encontrar pilotos similares", type="primary", use_container_width=True):

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
            <div style="padding:8px 14px; margin:4px 0; border-radius:8px; background:#2980b9;
                        border-left:3px solid #E10600;">
                <b>{row['piloto_id']}</b> ({row['year']}) ·
                <span style="color:#E10600; font-weight:600">{row['similitud_pct']}%</span> ·
                {row['archetype']}
            </div>""", unsafe_allow_html=True)

    with col_radar:
        st.subheader("🧬 Tu perfil vs. mejor match")
        mejor = df_matches.iloc[0]
        
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

    st.divider()
    st.subheader("📍 Posición en el espacio UMAP")
    st.caption(f"La estrella ★ muestra dónde cae tu Mejor Match ({mejor['piloto_id']}) para que veas tu Arquetipo en el mapa.")

    fig_sc = graficar_scatter_umap(df, highlight_driver=mejor['piloto_id'])
    st.plotly_chart(fig_sc, use_container_width=True)