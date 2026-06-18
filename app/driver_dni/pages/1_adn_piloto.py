import sys
from pathlib import Path
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_HERE         = Path(__file__).resolve().parent
_DNI          = _HERE.parent
_MODELS       = _DNI / "models"

sys.path.insert(0, str(_MODELS))
sys.path.insert(0, str(_DNI))

from similarity import (
    top_similares,
    graficar_barra_similares,
    FEATURE_COLS,
    FEATURE_LABELS,
)
from clustering import graficar_scatter_umap, ARCHETYPE_COLORS

MODELS_DIR    = _DNI
ARTIFACTS_DIR = _MODELS / "artifacts"

st.set_page_config(page_title="ADN del Piloto · F1 DNA", layout="wide")

def hex_to_rgba(hex_color: str, alpha: float = 0.2) -> str:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c*2 for c in h)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

@st.cache_data
def cargar_datos():
    # 🚨 ARREGLO 1: Leer el archivo FINAL para que las coordenadas UMAP sean idénticas al main.py
    ruta_final = MODELS_DIR / "driver_features_final.csv"
    if not ruta_final.exists():
        ruta_final = MODELS_DIR / "driver_features_clustered.csv"
        
    df = pd.read_csv(ruta_final)
    
    # 🚨 ARREGLO 2: Estandarizar la columna para que tu código de colores y gráficos no se rompa
    if "arquetipo" in df.columns and "archetype" not in df.columns:
        df = df.rename(columns={"arquetipo": "archetype"})
        
    return df

@st.cache_resource
def cargar_modelos():
    knn    = joblib.load(ARTIFACTS_DIR / "knn.pkl")
    return knn

df = cargar_datos()
knn = cargar_modelos()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🧬 ADN del Piloto")
st.markdown("Explorá el perfil de rendimiento de un piloto en una temporada específica.")

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

mask = (df["piloto_id"].astype(str) == piloto_elegido) & (df["year"] == year_elegido)
fila = df[mask].iloc[0] if mask.any() else None

if fila is None:
    st.error("No se encontraron datos para este piloto/temporada.")
    st.stop()

# ── Métricas (USAN COLUMNAS REALES) ───────────────────────────────────────────
st.divider()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("ELO pico",       f"{fila['elo_pico_temporada']:.0f}")
m2.metric("ELO final",      f"{fila['elo_final_temporada']:.0f}")
m3.metric("Win rate duelos",f"{fila['win_rate_duelos_carrera']*100:.1f}%")
m4.metric("Progresión ELO", f"{'+' if fila['elo_trend'] > 0 else ''}{fila['elo_trend']:.2f}")
m5.metric("Carreras",       int(fila.get("carreras", 0)))

archetype  = fila.get("archetype", "—")
color_arch = ARCHETYPE_COLORS.get(archetype, "#888888")
st.markdown(f"""
<div style="margin:12px 0; padding:10px 18px; border-radius:8px;
            background:{color_arch}22; border-left:4px solid {color_arch};">
    <b>Arquetipo:</b> <span style="color:{color_arch}; font-weight:600;">{archetype}</span>
</div>
""", unsafe_allow_html=True)

# ── Radar + similares ─────────────────────────────────────────────────────────
st.divider()
col_radar, col_similares = st.columns(2)

with col_radar:
    st.subheader("Radar de dimensiones")
    st.caption("Cada eje normalizado [0-1]. 'Consistencia': más alto = más consistente.")

    values = []
    # EL RADAR USA EXCLUSIVAMENTE LAS COLUMNAS _NORM PARA NO DEFORMARSE
    for f in FEATURE_COLS:
        col_norm = f"{f}_norm"
        v = float(fila[col_norm]) if pd.notnull(fila[col_norm]) else 0
        if f == "elo_volatilidad":
            v = 1 - v # Invertimos volatilidad para que visualmente sea "Consistencia"
        values.append(v)

    labels         = [FEATURE_LABELS[f] for f in FEATURE_COLS]
    values_closed  = values + [values[0]]
    labels_closed  = labels + [labels[0]]

    fig_radar = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=hex_to_rgba(color_arch, 0.2),
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
    st.caption("Similitud coseno sobre el vector de 7 features.")
    try:
        df_similares = top_similares(piloto_elegido, year_elegido, df, knn, n=5)
        fig_barras   = graficar_barra_similares(
            df_similares, titulo=f"Más parecidos a {piloto_elegido} ({year_elegido})"
        )
        st.plotly_chart(fig_barras, use_container_width=True)
    except ValueError as e:
        st.warning(str(e))

# ── Scatter UMAP ──────────────────────────────────────────────────────────────
st.divider()
st.subheader("📍 Posición en el espacio UMAP")
st.caption("Cada punto = un piloto en una temporada. Colores = arquetipos. ★ = piloto seleccionado.")
# Ahora esto leerá las coordenadas exactas de driver_features_final.csv
fig_scatter = graficar_scatter_umap(df, highlight_driver=piloto_elegido)
st.plotly_chart(fig_scatter, use_container_width=True)

# ── Evolución histórica (USA COLUMNAS REALES) ─────────────────────────────────
st.divider()
st.subheader("📈 Evolución del ELO a lo largo de su carrera")
df_hist = df[df["piloto_id"].astype(str) == piloto_elegido].sort_values("year")

if len(df_hist) > 1:
    fig_evol = go.Figure()
    fig_evol.add_trace(go.Scatter(
        x=df_hist["year"],
        y=df_hist["elo_final_temporada"], # Toma el dato real de la BD
        mode="lines+markers",
        name="ELO final por temporada",
        line=dict(color=color_arch, width=2.5),
        marker=dict(size=8),
        hovertemplate="<b>%{x}</b><br>ELO Final: %{y:.0f}<extra></extra>"
    ))
    # Ajusta aquí el 1000 o 1500 según cuál sea tu base configurada
    fig_evol.add_hline(y=1000, line_dash="dot", line_color="gray", annotation_text="ELO base (1000)")
    fig_evol.update_layout(
        xaxis_title="Temporada",
        yaxis_title="ELO final (Al cerrar el año)",
        template="plotly_white",
        height=320
    )
    st.plotly_chart(fig_evol, use_container_width=True)
else:
    st.info("Se necesitan al menos 2 temporadas válidas para mostrar la evolución.")