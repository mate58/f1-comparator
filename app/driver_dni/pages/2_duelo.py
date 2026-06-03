"""
pages/2_duelo.py
================
Página 2: Duelo de Pilotos
Compara dos pilotos/temporadas con radar superpuesto, similitud global
y desglose por dimensión.
"""

import sys
import numpy as np
from pathlib import Path

_HERE    = Path(__file__).resolve().parent
_DNI     = _HERE.parent
_MODELS  = _DNI / "models"

sys.path.insert(0, str(_MODELS))
sys.path.insert(0, str(_DNI))

MODELS_DIR = _DNI
ARTIFACTS_DIR = _MODELS / "artifacts"

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from similarity import FEATURE_COLS, FEATURE_LABELS
from clustering import ARCHETYPE_COLORS

st.set_page_config(page_title="Duelo de Pilotos · F1 DNA", layout="wide")

@st.cache_data
def cargar_datos():
    return pd.read_csv(MODELS_DIR / "driver_features_clustered.csv")

df = cargar_datos()
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]

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
    # Extraemos filas
    fila_a = df[(df["piloto_id"].astype(str)==piloto_a) & (df["year"]==year_a)]
    fila_b = df[(df["piloto_id"].astype(str)==piloto_b) & (df["year"]==year_b)]
    
    if fila_a.empty or fila_b.empty:
        st.error("No hay datos para esta combinación.")
    else:
        fila_a = fila_a.iloc[0]
        fila_b = fila_b.iloc[0]

        # 1. Similitud global (Coseno sobre _norm)
        vec_a = fila_a[COLS_NORM].fillna(0).values.astype(float)
        vec_b = fila_b[COLS_NORM].fillna(0).values.astype(float)
        
        # Distancia Coseno manual segura
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            sim_pct = 0.0
        else:
            cos_sim = np.dot(vec_a, vec_b) / (norm_a * norm_b)
            sim_pct = round(max(0.0, float(cos_sim) * 100), 1)

        # UI Score
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
        arch_a = fila_a.get("archetype","—")
        arch_b = fila_b.get("archetype","—")
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

        # Radar y Deltas
        col_r, col_d = st.columns([1, 1])
        
        # Invertimos volatilidad para el radar ("Consistencia")
        valores_a = []
        valores_b = []
        deltas = []
        
        for i, col in enumerate(FEATURE_COLS):
            val_a = vec_a[i]
            val_b = vec_b[i]
            if col == "elo_volatilidad":
                val_a = 1.0 - val_a
                val_b = 1.0 - val_b
                
            valores_a.append(val_a)
            valores_b.append(val_b)
            deltas.append(val_a - val_b) # Delta visual (A - B)

        labels = [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS]

        with col_r:
            st.subheader("Radar superpuesto")
            fig_r = go.Figure()
            fig_r.add_trace(go.Scatterpolar(
                r=valores_a + [valores_a[0]], theta=labels + [labels[0]], fill="toself",
                name=f"🔴 {piloto_a}", fillcolor="rgba(225,6,0,0.15)", line=dict(color="#E10600", width=2)
            ))
            fig_r.add_trace(go.Scatterpolar(
                r=valores_b + [valores_b[0]], theta=labels + [labels[0]], fill="toself",
                name=f"🔵 {piloto_b}", fillcolor="rgba(0,144,208,0.15)", line=dict(color="#0090D0", width=2)
            ))
            fig_r.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True, template="plotly_white", height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col_d:
            st.subheader("Diferencia por dimensión")
            st.caption("Positivo → Piloto A supera a Piloto B en esa dimensión.")
            
            df_delta = pd.DataFrame({"Dimensión": labels, "Delta": deltas})
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
                xaxis_title="Delta normalizado (A - B)",
                yaxis=dict(autorange="reversed"),
                template="plotly_white",
                height=380, margin=dict(t=10)
            )
            st.plotly_chart(fig_delta, use_container_width=True)

        # ── Época de mayor parecido ─────────────────────────────────────────
        st.divider()
        st.subheader(f"📅 ¿Cuándo fueron más parecidos {piloto_a} y {piloto_b}?")

        df_a_hist = df[df["piloto_id"].astype(str) == piloto_a]
        df_b_hist = df[df["piloto_id"].astype(str) == piloto_b]
        years_comun = sorted(list(set(df_a_hist["year"]).intersection(set(df_b_hist["year"]))))

        if years_comun:
            sims_historicas = []
            for yr in years_comun:
                va = df_a_hist[df_a_hist["year"]==yr][COLS_NORM].fillna(0).values.astype(float)[0]
                vb = df_b_hist[df_b_hist["year"]==yr][COLS_NORM].fillna(0).values.astype(float)[0]
                
                na, nb = np.linalg.norm(va), np.linalg.norm(vb)
                if na > 0 and nb > 0:
                    csim = np.dot(va, vb) / (na * nb)
                    sims_historicas.append({"year": yr, "similitud_pct": round(max(0.0, csim*100), 1)})

            df_hist_sim = pd.DataFrame(sims_historicas)
            fig_hist = go.Figure(go.Scatter(
                x=df_hist_sim["year"], y=df_hist_sim["similitud_pct"], mode="lines+markers",
                line=dict(color="#E10600", width=2), marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Similitud: %{y:.1f}%<extra></extra>"
            ))
            fig_hist.update_layout(
                xaxis_title="Temporada", yaxis_title="Similitud (%)",
                yaxis=dict(range=[0, 100]), template="plotly_white", height=280
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No coincidieron en las mismas temporadas para comparar evolución histórica.")