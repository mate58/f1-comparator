"""
main.py — Entry point del dashboard F1 Driver DNA
==================================================
Streamlit multi-página. Ejecutar con:
  streamlit run app/main.py
"""

import streamlit as st

st.set_page_config(
    page_title="F1 Driver DNA",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background-color: #1a1a2e; }
  [data-testid="stSidebar"] * { color: #f0f0f0 !important; }
  .metric-card {
    background: #f8f9fa; border-radius: 12px;
    padding: 16px 20px; margin: 8px 0;
    border-left: 4px solid #E10600;
  }
  .similitud-badge {
    display: inline-block; padding: 4px 12px;
    border-radius: 20px; font-weight: 600;
    background: #E10600; color: white; font-size: 0.9rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/33/F1.svg/200px-F1.svg.png", width=120)
    st.markdown("## 🏎️ F1 Driver DNA")
    st.markdown("Aprendizaje no supervisado aplicado a datos históricos de Fórmula 1 (2000–2024)")
    st.divider()
    st.markdown("### Navegación")
    st.page_link("pages/1_adn_piloto.py",    label="🧬 ADN del Piloto",       icon="🧬")
    st.page_link("pages/2_duelo.py",          label="⚔️  Duelo de Pilotos",    icon="⚔️")
    st.page_link("pages/3_piloto_custom.py",  label="🎛️  Piloto Hipotético",   icon="🎛️")
    st.divider()
    st.caption("Metodología: UMAP + K-Means + Similitud Coseno")
    st.caption("Datos: Ergast API · OpenF1 API")

# ── Home ───────────────────────────────────────────────────────────────────────
st.title("🏎️ F1 Driver DNA")
st.subheader("¿A qué piloto histórico te parecés?")

st.markdown("""
Este proyecto aplica **aprendizaje no supervisado** sobre datos históricos de Fórmula 1
para construir un sistema de comparación y clasificación de pilotos.

El sistema asigna a cada piloto (por temporada) un **vector de features** derivado de
su sistema ELO propio, ajustado por calidad del auto y rendimiento vs. compañero de escudería.
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>🧬 ADN del Piloto</h3>
        <p>Ingresá un piloto y temporada para ver su perfil multidimensional,
        a qué arquetipo pertenece y cuáles son los 5 pilotos históricos más similares.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <h3>⚔️ Duelo de Pilotos</h3>
        <p>Compará dos pilotos cara a cara. El sistema muestra similitud global,
        diferencias por dimensión y en qué época fueron más parecidos.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <h3>🎛️ Piloto Hipotético</h3>
        <p>Definí un piloto ideal con sliders (consistencia, rendimiento vs. compañero,
        calidad del auto, etc.) y descubrí a quién se parece en la historia de la F1.</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# Métricas globales del dataset
st.subheader("📊 Dataset")
mc1, mc2, mc3, mc4 = st.columns(4)
mc1.metric("Temporadas", "2000 – 2024")
mc2.metric("Pilotos únicos", "~120")
mc3.metric("Vectores piloto/temporada", "~800")
mc4.metric("Features por vector", "9")