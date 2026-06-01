"""
pages/2_duelo.py
================
Página 2: Duelo de Pilotos
Compara dos pilotos/temporadas con radar superpuesto, similitud global
y desglose por dimensión.
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

from similarity import (
    comparar_dos_pilotos,
    graficar_radar,
    FEATURE_COLS,
    FEATURE_LABELS,
)
from clustering import ARCHETYPE_COLORS

st.set_page_config(page_title="Duelo de Pilotos · F1 DNA", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv(MODELS_DIR / "driver_features_clustered.csv")

df = cargar_datos()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("⚔️ Duelo de Pilotos")
st.markdown("Compará dos pilotos cara a cara para ver qué tan similares son y en qué dimensiones difieren.")

# ── Selectores lado a lado ────────────────────────────────────────────────────
col_a, col_versus, col_b = st.columns([5, 1, 5])
pilotos = sorted(df["piloto_id"].astype(str).unique())

with col_a:
    st.markdown("#### 🔴 Piloto A")
    piloto_a = st.selectbox("Piloto A", pilotos, key="pa", index=0)
    years_a  = sorted(df[df["piloto_id"].astype(str) == piloto_a]["year"].unique(), reverse=True)
    year_a   = st.selectbox("Temporada A", years_a, key="ya")

with col_versus:
    st.markdown("<br><br><h2 style='text-align:center'>VS</h2>", unsafe_allow_html=True)

with col_b:
    st.markdown("#### 🔵 Piloto B")
    piloto_b = st.selectbox("Piloto B", pilotos, key="pb", index=min(5, len(pilotos)-1))
    years_b  = sorted(df[df["piloto_id"].astype(str) == piloto_b]["year"].unique(), reverse=True)
    year_b   = st.selectbox("Temporada B", years_b, key="yb")

st.divider()

# ── Comparación ───────────────────────────────────────────────────────────────
if st.button("⚔️ Comparar", type="primary", use_container_width=True):
    try:
        resultado = comparar_dos_pilotos(piloto_a, year_a, piloto_b, year_b, df)
        sim_pct   = resultado["similitud_global_pct"]
        delta     = resultado["delta_por_feature"]
        radar     = resultado["radar_data"]

        # Score de similitud global
        color_sim = "#00C851" if sim_pct >= 75 else "#FF8700" if sim_pct >= 50 else "#E10600"
        st.markdown(f"""
        <div style="text-align:center; padding: 20px; background: {color_sim}18;
                    border-radius:12px; border: 2px solid {color_sim};">
            <h1 style="color:{color_sim}; margin:0;">{sim_pct}%</h1>
            <p style="color:#555; margin:4px 0 0;">Similitud global (coseno)</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")

        # Arquetipos
        fila_a = df[(df["piloto_id"].astype(str)==piloto_a) & (df["year"]==year_a)]
        fila_b = df[(df["piloto_id"].astype(str)==piloto_b) & (df["year"]==year_b)]
        arch_a = fila_a.iloc[0].get("archetype","—") if not fila_a.empty else "—"
        arch_b = fila_b.iloc[0].get("archetype","—") if not fila_b.empty else "—"
        col1, col2 = st.columns(2)
        with col1:
            ca = ARCHETYPE_COLORS.get(arch_a, "#888")
            st.markdown(f"""
            <div style="padding:10px 16px; border-radius:8px;
                        background:{ca}18; border-left:4px solid {ca}; text-align:center">
                <b>{piloto_a} ({year_a})</b><br>
                <span style="color:{ca}; font-weight:600">{arch_a}</span>
            </div>""", unsafe_allow_html=True)
        with col2:
            cb = ARCHETYPE_COLORS.get(arch_b, "#888")
            st.markdown(f"""
            <div style="padding:10px 16px; border-radius:8px;
                        background:{cb}18; border-left:4px solid {cb}; text-align:center">
                <b>{piloto_b} ({year_b})</b><br>
                <span style="color:{cb}; font-weight:600">{arch_b}</span>
            </div>""", unsafe_allow_html=True)

        st.divider()

        # Radar superpuesto
        col_r, col_d = st.columns([1, 1])

        with col_r:
            st.subheader("Radar superpuesto")
            fig_radar = graficar_radar(radar)
            st.plotly_chart(fig_radar, use_container_width=True)

        with col_d:
            st.subheader("Diferencia por dimensión")
            st.caption("Positivo → Piloto A supera a Piloto B en esa dimensión.")

            rows = []
            for feat, diff in delta.items():
                label = FEATURE_LABELS.get(feat, feat)
                ganador = f"🔴 {piloto_a}" if diff > 0.02 else (f"🔵 {piloto_b}" if diff < -0.02 else "Empate")
                rows.append({
                    "Dimensión" : label,
                    "Delta"     : round(diff, 3),
                    "Ventaja"   : ganador
                })

            df_delta = pd.DataFrame(rows)

            fig_delta = go.Figure(go.Bar(
                x=df_delta["Delta"],
                y=df_delta["Dimensión"],
                orientation="h",
                marker_color=[
                    "#E10600" if d > 0.02 else "#0090D0" if d < -0.02 else "#AAAAAA"
                    for d in df_delta["Delta"]
                ],
                text=[f"{d:+.3f}" for d in df_delta["Delta"]],
                textposition="outside",
            ))
            fig_delta.add_vline(x=0, line_dash="dot", line_color="gray")
            fig_delta.update_layout(
                xaxis_title="Delta (A - B)",
                yaxis=dict(autorange="reversed"),
                template="plotly_white",
                height=380,
                margin=dict(t=10)
            )
            st.plotly_chart(fig_delta, use_container_width=True)

        # ── Época de mayor parecido ─────────────────────────────────────────
        st.divider()
        st.subheader(f"📅 ¿Cuándo fueron más parecidos {piloto_a} y {piloto_b}?")

        df_a_hist = df[df["piloto_id"].astype(str) == piloto_a]
        df_b_hist = df[df["piloto_id"].astype(str) == piloto_b]
        years_comun = set(df_a_hist["year"]).intersection(set(df_b_hist["year"]))

        if years_comun:
            sims_historicas = []
            for yr in sorted(years_comun):
                try:
                    r = comparar_dos_pilotos(piloto_a, yr, piloto_b, yr, df)
                    sims_historicas.append({"year": yr, "similitud_pct": r["similitud_global_pct"]})
                except ValueError:
                    pass

            if sims_historicas:
                df_hist_sim = pd.DataFrame(sims_historicas)
                fig_hist = go.Figure(go.Scatter(
                    x=df_hist_sim["year"],
                    y=df_hist_sim["similitud_pct"],
                    mode="lines+markers",
                    line=dict(color="#E10600", width=2),
                    marker=dict(size=8),
                    hovertemplate="<b>%{x}</b><br>Similitud: %{y:.1f}%<extra></extra>"
                ))
                fig_hist.update_layout(
                    xaxis_title="Temporada",
                    yaxis_title="Similitud (%)",
                    yaxis=dict(range=[0, 100]),
                    template="plotly_white",
                    height=280
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No coincidieron en las mismas temporadas para comparar evolución histórica.")

    except ValueError as e:
        st.error(str(e))

else:
    st.info("Seleccioná ambos pilotos y hacé clic en **Comparar** para ver el análisis completo.")