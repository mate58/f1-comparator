"""
pages/1_adn_piloto.py
=====================
Página 1: ADN del Piloto
Muestra el perfil multidimensional de un piloto en una temporada específica,
el arquetipo al que pertenece y los 5 pilotos históricos más similares.
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
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path

from similarity import (
    top_similares,
    comparar_dos_pilotos,
    graficar_radar,
    graficar_barra_similares,
    FEATURE_COLS,
    FEATURE_LABELS,
)
from clustering import graficar_scatter_umap, ARCHETYPE_COLORS

st.set_page_config(page_title="ADN del Piloto · F1 DNA", layout="wide")

# ── Carga de datos ────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    df = pd.read_csv(MODELS_DIR / "driver_features_clustered.csv")
    return df

@st.cache_resource
def cargar_modelos():
    knn    = joblib.load(ARTIFACTS_DIR / "knn.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    return knn, scaler

df = cargar_datos()
knn, scaler = cargar_modelos()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 ADN del Piloto")
st.markdown("Explorá el perfil de rendimiento de un piloto en una temporada específica.")

# ── Selector ─────────────────────────────────────────────────────────────────
col_sel1, col_sel2 = st.columns([2, 1])
with col_sel1:
    pilotos_disponibles = sorted(df["piloto_id"].astype(str).unique())
    piloto_elegido = st.selectbox("Piloto", pilotos_disponibles, index=0)

with col_sel2:
    years_piloto = sorted(
        df[df["piloto_id"].astype(str) == piloto_elegido]["year"].unique(),
        reverse=True
    )
    year_elegido = st.selectbox("Temporada", years_piloto)

# ── Obtener datos del piloto ──────────────────────────────────────────────────
mask = (df["piloto_id"].astype(str) == piloto_elegido) & (df["year"] == year_elegido)
fila = df[mask].iloc[0] if mask.any() else None

if fila is None:
    st.error("No se encontraron datos para este piloto/temporada.")
    st.stop()

# ── Métricas rápidas ──────────────────────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ELO pico",       f"{fila['elo_pico_temporada']:.0f}")
m2.metric("ELO final",      f"{fila['elo_final_temporada']:.0f}")
m3.metric("Win rate duelos",f"{fila['win_rate_duelos_carrera']*100:.1f}%")
m4.metric("Progresión ELO", f"{'+' if fila['elo_trend'] > 0 else ''}{fila['elo_trend']:.2f}")
m5.metric("Carreras",       int(fila.get("carreras", 0)))

archetype = fila.get("archetype", "—")
color_arch = ARCHETYPE_COLORS.get(archetype, "#888")
st.markdown(f"""
<div style="margin:12px 0; padding:10px 18px; border-radius:8px;
            background:{color_arch}22; border-left:4px solid {color_arch};">
    <b>Arquetipo:</b> <span style="color:{color_arch}; font-weight:600;">{archetype}</span>
</div>
""", unsafe_allow_html=True)

# ── Radar chart ───────────────────────────────────────────────────────────────
st.divider()
col_radar, col_similares = st.columns([1, 1])

with col_radar:
    st.subheader("Radar de dimensiones")
    st.caption("Cada eje es un feature normalizado [0-1]. En 'Consistencia': más alto = más consistente.")

    # Invertir volatilidad para que radar sea "más = mejor"
    values = []
    for f in FEATURE_COLS:
        v = float(fila[f]) if pd.notnull(fila[f]) else 0
        if f == "elo_volatilidad":
            v = 1 - v  # invertir: baja volatilidad = alta consistencia
        values.append(v)

    labels = [FEATURE_LABELS[f] for f in FEATURE_COLS]
    values_closed = values + [values[0]]
    labels_closed = labels + [labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=f"{color_arch}33",
        line=dict(color=color_arch, width=2.5),
        name=f"{piloto_elegido} ({year_elegido})"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        template="plotly_white",
        height=400,
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

with col_similares:
    st.subheader("Top 5 pilotos más similares")
    st.caption("Calculado con similitud coseno sobre el vector de 9 features.")

    try:
        df_similares = top_similares(piloto_elegido, year_elegido, df, knn, n=5)
        fig_barras = graficar_barra_similares(
            df_similares, titulo=f"Más parecidos a {piloto_elegido} ({year_elegido})"
        )
        st.plotly_chart(fig_barras, use_container_width=True)

        # Tabla detallada
        st.dataframe(
            df_similares.rename(columns={
                "piloto_id"    : "Piloto",
                "year"         : "Año",
                "similitud_pct": "Similitud %",
                "archetype"    : "Arquetipo",
            }),
            use_container_width=True,
            hide_index=True
        )
    except ValueError as e:
        st.warning(str(e))

# ── Scatter UMAP ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("📍 Posición en el espacio UMAP")
st.caption(
    "Cada punto es un piloto en una temporada. Los colores representan arquetipos. "
    "El piloto seleccionado aparece como ★."
)

fig_scatter = graficar_scatter_umap(df, highlight_driver=piloto_elegido)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Evolución histórica del ELO ───────────────────────────────────────────────
st.divider()
st.subheader("📈 Evolución del ELO a lo largo de su carrera")
df_piloto_hist = df[df["piloto_id"].astype(str) == piloto_elegido].sort_values("year")

if len(df_piloto_hist) > 1:
    fig_evol = go.Figure()
    fig_evol.add_trace(go.Scatter(
        x=df_piloto_hist["year"],
        y=df_piloto_hist["elo_final_temporada"],
        mode="lines+markers",
        name="ELO final por temporada",
        line=dict(color=color_arch, width=2.5),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>ELO: %{y:.0f}<extra></extra>"
    ))
    fig_evol.add_hline(y=1000, line_dash="dot", line_color="gray", annotation_text="ELO base (1000)")
    fig_evol.update_layout(
        xaxis_title="Temporada",
        yaxis_title="ELO final",
        template="plotly_white",
        height=320
    )
    st.plotly_chart(fig_evol, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 temporadas para mostrar la evolución.")