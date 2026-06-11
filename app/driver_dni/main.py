"""
main.py — Entry point del dashboard F1 Driver DNA
==================================================
Streamlit multi-página. Ejecutar con:
  streamlit run main.py
"""

import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="F1 Driver DNA",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global y estilos ────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stSidebar"] { background-color: #15151e; border-right: 1px solid #2b2b36; }
    .nav-card {
        background-color: #1e1e28; border-radius: 12px; padding: 24px;
        margin: 10px 0; border-left: 5px solid #E10600; color: white;
    }
    .badge {
        display: inline-block; padding: 6px 12px;
        border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        margin-right: 8px; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Funciones de Datos ────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos():
    """Lee el CSV final generado por el backend."""
    rutas = [
        Path(__file__).resolve().parent / "models" / "driver_features_final.csv",
        Path(__file__).resolve().parent / "driver_features_final.csv"
    ]
    for ruta in rutas:
        if ruta.exists():
            return pd.read_csv(ruta)
    return None

df_final = cargar_datos()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align:center; font-size: 4rem; margin-bottom: 0;'>🏎️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-top: 0;'>Driver DNA</h2>", unsafe_allow_html=True)
    st.divider()
    st.caption("⚙️ **Metodología:** UMAP + K-Means + KNN")
    st.caption("📊 **Datos:** Basado en historial ELO")

# ── Home ───────────────────────────────────────────────────────────────────────
st.title("🏎️ Proyecto F1 Driver DNA")
st.subheader("Decodificando el rendimiento de los pilotos con Machine Learning")
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='nav-card'><h3>🧬 Pantalla 1: ADN del Piloto</h3><p>Buscá un piloto y descubrí sus gemelos históricos.</p></div>", unsafe_allow_html=True)
with col2:
    st.markdown("<div class='nav-card'><h3>⚔️ Pantalla 2: Duelo</h3><p>Enfrentá a dos pilotos cara a cara.</p></div>", unsafe_allow_html=True)
with col3:
    st.markdown("<div class='nav-card'><h3>🎛️ Pantalla 3: Hipotético</h3><p>Definí tu perfil ideal y encontrá coincidencias.</p></div>", unsafe_allow_html=True)

st.divider()

# ── SECCIÓN PARA LA DEFENSA DEL PROYECTO: PCA vs UMAP ─────────────────────────
if df_final is not None and 'pca_x' in df_final.columns and 'umap_x' in df_final.columns:
    st.markdown("### 🔬 Defensa Técnica: Evaluación de Reducción de Dimensionalidad")
    st.markdown("Comparativa entre el enfoque estadístico clásico lineal y el mapeo topológico no-lineal.")
    
    col_pca, col_umap = st.columns(2)
    
    # Mapeo de colores para mantener coherencia visual
    colores_arquetipos = {
        "El Dominador Absoluto": "#E10600", "El Maestro a Una Vuelta": "#f39c12",
        "El Muro Defensivo": "#27ae60", "El Piloto Escudero": "#2980b9",
        "El Sobreviviente": "#7f8c8d"
    }

    # FIX de idioma de columna
    col_arq = "arquetipo" if "arquetipo" in df_final.columns else "archetype"

    with col_pca:
        st.markdown("#### 1. Método PCA (Baseline Lineal)")
        fig_pca = px.scatter(
            df_final, x='pca_x', y='pca_y', color=col_arq,
            color_discrete_map=colores_arquetipos,
            hover_data=['piloto_id', 'year'],
            labels={'pca_x': 'Componente 1', 'pca_y': 'Componente 2'}
        )
        fig_pca.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_umap:
        st.markdown("#### 2. Método UMAP (Topología No-Lineal)")
        fig_umap = px.scatter(
            df_final, x='umap_x', y='umap_y', color=col_arq,
            color_discrete_map=colores_arquetipos,
            hover_data=['piloto_id', 'year'],
            labels={'umap_x': 'Dimensión UMAP 1', 'umap_y': 'Dimensión UMAP 2'}
        )
        fig_umap.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_umap, use_container_width=True)
else:
    st.warning("Ejecuta `dimensionality.py` en tu terminal para generar los datos de la comparativa.")