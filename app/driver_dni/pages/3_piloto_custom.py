"""
pages/3_piloto_custom.py
========================
Página 3: Piloto Hipotético
El usuario define un perfil con sliders y el modelo encuentra los pilotos
históricos más similares.

Esta página demuestra que el modelo GENERALIZA: puede mapear un vector
arbitrario al espacio UMAP entrenado y encontrar vecinos históricos.
Esto no sería posible con t-SNE (no soporta transform() sobre nuevos puntos).
"""

import sys
from pathlib import Path

# pages/ está en driver_dni/pages/
# Los módulos (similarity.py, clustering.py, etc.) están en driver_dni/models/
# Los CSVs y PKLs también están en driver_dni/ (build_features los deja ahí)
_HERE    = Path(__file__).resolve().parent        # driver_dni/pages/
_DNI     = _HERE.parent                           # driver_dni/
_MODELS  = _DNI / "models"                        # driver_dni/models/

sys.path.insert(0, str(_MODELS))   # para: from similarity import ...
sys.path.insert(0, str(_DNI))      # por si acaso

MODELS_DIR = _DNI   # CSVs y scaler.pkl viven en driver_dni/
ARTIFACTS_DIR = _MODELS / "artifacts"   # PKLs viven en driver_dni/models/artifacts/


import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from similarity import (
    match_piloto_hipotetico,
    graficar_barra_similares,
    graficar_radar,
    comparar_dos_pilotos,
    FEATURE_COLS,
    FEATURE_LABELS,
)
from dimensionality import proyectar_nuevo_piloto
from clustering import graficar_scatter_umap

st.set_page_config(page_title="Piloto Hipotético · F1 DNA", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv(MODELS_DIR / "driver_features_clustered.csv")

@st.cache_resource
def cargar_modelos():
    knn    = joblib.load(ARTIFACTS_DIR / "knn.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return knn, scaler

df     = cargar_datos()
knn, scaler = cargar_modelos()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎛️ Piloto Hipotético")
st.markdown("""
Definí el perfil de un piloto usando los sliders. El sistema proyectará ese perfil
en el espacio UMAP y encontrará los pilotos históricos más parecidos.

> **Tip:** Usá los **presets** para explorar perfiles conocidos, o construí el tuyo desde cero.
""")

# ── Presets ───────────────────────────────────────────────────────────────────
st.markdown("#### Presets rápidos")
preset_cols = st.columns(5)
presets = {
    "Dominador"    : {"elo_pico":0.95, "elo_final":0.93, "volatilidad":0.1, "qualy_race":0.6,
                      "companero":0.85, "win_rate":0.90, "car_bonus":0.95, "trend":0.5, "remontada":0.5},
    "Sobre-rendidor":{"elo_pico":0.60, "elo_final":0.58, "volatilidad":0.3, "qualy_race":0.7,
                      "companero":0.75, "win_rate":0.78, "car_bonus":0.20, "trend":0.55, "remontada":0.80},
    "Clasificador" : {"elo_pico":0.70, "elo_final":0.65, "volatilidad":0.2, "qualy_race":0.2,
                      "companero":0.55, "win_rate":0.60, "car_bonus":0.60, "trend":0.4, "remontada":0.35},
    "Consistente"  : {"elo_pico":0.55, "elo_final":0.54, "volatilidad":0.05,"qualy_race":0.5,
                      "companero":0.50, "win_rate":0.55, "car_bonus":0.50, "trend":0.1, "remontada":0.45},
    "Rookie"       : {"elo_pico":0.35, "elo_final":0.40, "volatilidad":0.7, "qualy_race":0.4,
                      "companero":0.30, "win_rate":0.35, "car_bonus":0.50, "trend":0.90, "remontada":0.50},
}

selected_preset = None
for col, (nombre, vals) in zip(preset_cols, presets.items()):
    with col:
        if st.button(nombre, use_container_width=True):
            selected_preset = vals

# ── Sliders ───────────────────────────────────────────────────────────────────
st.divider()
st.markdown("#### Configurá el perfil")
st.caption("Todos los valores están normalizados en [0, 1]. El modelo los mapea al espacio de features del historial ELO.")

def get_default(key: str, presets_vals: dict | None, field: str, default: float) -> float:
    return presets_vals.get(field, default) if presets_vals else default

c1, c2, c3 = st.columns(3)

with c1:
    elo_pico    = st.slider("ELO pico temporada",      0.0, 1.0,
                            get_default("elo_pico", selected_preset, "elo_pico", 0.5), 0.01,
                            help="Qué tan alto llegó su ELO en la temporada (0=base, 1=máximo histórico)")
    elo_final   = st.slider("ELO final temporada",     0.0, 1.0,
                            get_default("elo_final", selected_preset, "elo_final", 0.48), 0.01,
                            help="ELO al terminar la última carrera")
    volatilidad = st.slider("Consistencia (inverso volatilidad)", 0.0, 1.0,
                            get_default("volatilidad", selected_preset, "volatilidad", 0.5), 0.01,
                            help="Alto = muy consistente. Bajo = resultados erráticos")

with c2:
    qualy_race  = st.slider("Mejora qualy → carrera",  0.0, 1.0,
                            get_default("qualy_race", selected_preset, "qualy_race", 0.5), 0.01,
                            help="Alto = piloto que mejora del sábado al domingo")
    companero   = st.slider("Delta vs. compañero",     0.0, 1.0,
                            get_default("companero", selected_preset, "companero", 0.5), 0.01,
                            help="Alto = le gana consistentemente al compañero de escudería")
    win_rate    = st.slider("Win rate duelos",          0.0, 1.0,
                            get_default("win_rate", selected_preset, "win_rate", 0.5), 0.01,
                            help="% de duelos ELO ganados en carrera")

with c3:
    car_bonus   = st.slider("Calidad del auto",         0.0, 1.0,
                            get_default("car_bonus", selected_preset, "car_bonus", 0.5), 0.01,
                            help="Alto = auto top de la parrilla. Bajo = equipo de midfield o fondo")
    trend       = st.slider("Progresión en la temporada",0.0, 1.0,
                            get_default("trend", selected_preset, "trend", 0.5), 0.01,
                            help="Alto = fue mejorando a lo largo del año")
    remontada   = st.slider("Tasa de remontada",        0.0, 1.0,
                            get_default("remontada", selected_preset, "remontada", 0.5), 0.01,
                            help="% de carreras donde ganó posiciones respecto a la salida")

# ── Construir vector ──────────────────────────────────────────────────────────
vector_custom = {
    "elo_pico_temporada"     : elo_pico,
    "elo_final_temporada"    : elo_final,
    "elo_volatilidad"        : 1 - volatilidad,   # Re-invertir para el modelo
    "elo_qualy_vs_race_delta": qualy_race,
    "elo_delta_companero"    : companero,
    "win_rate_duelos_carrera": win_rate,
    "car_bonus_avg"          : car_bonus,
    "elo_trend"              : trend,
    "remontada_rate"         : remontada,
}

st.divider()

# ── Buscar matches ────────────────────────────────────────────────────────────
if st.button("🔍 Encontrar pilotos similares", type="primary", use_container_width=True):

    df_matches = match_piloto_hipotetico(
        vector_custom, df, scaler, knn, n=5
    )

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
        try:
            resultado = comparar_dos_pilotos(
                mejor["piloto_id"], int(mejor["year"]),
                mejor["piloto_id"], int(mejor["year"]),  # placeholder
                df
            )
            # Radar del perfil custom vs. mejor match
            labels = [FEATURE_LABELS[f] for f in FEATURE_COLS]
            vec_custom_radar = [
                vector_custom[f] if f != "elo_volatilidad"
                else 1 - vector_custom[f]
                for f in FEATURE_COLS
            ]

            fila_match = df[
                (df["piloto_id"].astype(str) == str(mejor["piloto_id"])) &
                (df["year"] == int(mejor["year"]))
            ].iloc[0]

            vec_match_radar = []
            for f in FEATURE_COLS:
                v = float(fila_match[f]) if pd.notnull(fila_match[f]) else 0
                if f == "elo_volatilidad":
                    v = 1 - v
                vec_match_radar.append(v)

            cats = labels + [labels[0]]
            vc = vec_custom_radar + [vec_custom_radar[0]]
            vm = vec_match_radar  + [vec_match_radar[0]]

            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=vc, theta=cats, fill="toself", name="Tu piloto",
                fillcolor="rgba(225,6,0,0.15)",
                line=dict(color="#E10600", width=2)
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=vm, theta=cats, fill="toself",
                name=f"{mejor['piloto_id']} ({mejor['year']})",
                fillcolor="rgba(0,144,208,0.15)",
                line=dict(color="#0090D0", width=2)
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,1])),
                showlegend=True,
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_r, use_container_width=True)
        except Exception:
            st.info("No se pudo renderizar el radar comparativo.")

    # ── Scatter UMAP con piloto hipotético proyectado ──────────────────────
    st.divider()
    st.subheader("📍 Tu piloto en el espacio UMAP")
    st.caption("La estrella ★ muestra dónde caería tu perfil hipotético en el mapa de todos los pilotos históricos.")

    try:
        vec_array = np.array([[vector_custom[f] for f in FEATURE_COLS]])
        coords_2d = proyectar_nuevo_piloto(vec_array)

        fig_sc = graficar_scatter_umap(df)
        fig_sc.add_trace(go.Scatter(
            x=[coords_2d[0, 0]],
            y=[coords_2d[0, 1]],
            mode="markers+text",
            name="★ Tu piloto",
            marker=dict(size=18, color="#FFD700", symbol="star",
                        line=dict(width=2, color="#E10600")),
            text=["Tu piloto"],
            textposition="top center"
        ))
        st.plotly_chart(fig_sc, use_container_width=True)
    except Exception:
        st.info("Instalá umap-learn para proyectar el piloto hipotético en el mapa UMAP.")

else:
    st.info("Configurá el perfil con los sliders y hacé clic en **Encontrar pilotos similares**.")