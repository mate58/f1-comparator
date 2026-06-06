"""
main.py
==================================================
Streamlit multi-página. Ejecutar con:
  streamlit run main.py
"""

import pandas as pd
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="F1 Driver ADN",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS global y estilos mejorados ────────────────────────────────────────────
st.markdown("""
<style>
    /* Estilos del Sidebar */
    [data-testid="stSidebar"] { 
        background-color: #15151e; 
        border-right: 1px solid #2b2b36;
    }
    
    /* Tarjetas de Navegación Home */
    .nav-card {
        background-color: #1e1e28;
        border-radius: 12px;
        padding: 24px;
        margin: 10px 0;
        border-left: 5px solid #E10600;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        color: white;
    }
    .nav-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 16px rgba(225, 6, 0, 0.15);
    }
    .nav-card h3 { margin-top: 0; color: #ffffff; font-size: 1.4rem;}
    .nav-card p { color: #a0a0b0; font-size: 1rem; line-height: 1.5; }
    
    /* Badges de Arquetipos sincronizados con clustering.py */
    .badge {
        display: inline-block; padding: 6px 12px;
        border-radius: 20px; font-weight: 600; font-size: 0.85rem;
        margin-right: 8px; margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Funciones de Datos ────────────────────────────────────────────────────────
@st.cache_data
def obtener_metricas_dataset():
    """Lee el CSV generado para extraer estadísticas reales del proyecto."""
    
    # Intentamos leer el archivo con UMAP primero, si no existe vamos al clustered
    csv_path = Path(__file__).resolve().parent / "models" / "driver_features_final.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "driver_features_final.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "models" / "driver_features_clustered.csv"
    if not csv_path.exists():
        csv_path = Path(__file__).resolve().parent / "driver_features_clustered.csv"
        
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        
        # Leemos la columna 'arquetipo' (español) o 'archetype' (inglés) según lo que haya generado el modelo
        col_arquetipo = "arquetipo" if "arquetipo" in df.columns else "archetype"
        
        return {
            "pilotos": df["piloto_id"].nunique(),
            "vectores": len(df),
            "arquetipos": df[col_arquetipo].nunique() if col_arquetipo in df.columns else 0,
            "periodo": f"{df['year'].min()} – {df['year'].max()}",
            "dimensiones": 7
        }
    return None

stats = obtener_metricas_dataset()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h1 style='text-align:center; font-size: 4rem; margin-bottom: 0; padding-bottom: 0;'>🏎️</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center; margin-top: 0; padding-top: 0;'>Driver ADN</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#888;'>Análisis de perfiles con IA</p>", unsafe_allow_html=True)
    
    st.divider()
    
    st.caption("⚙️ **Metodología:** UMAP + K-Means + KNN")
    st.caption("📊 **Datos:** Basado en historial ELO")

# ── Home ───────────────────────────────────────────────────────────────────────
st.title("🏎️ Proyecto F1 Driver ADN")
st.subheader("Decodificando el rendimiento de los pilotos con Machine Learning")

st.markdown("""
Este dashboard utiliza **aprendizaje no supervisado** para analizar la historia de la Fórmula 1. 
En lugar de mirar simples estadísticas de victorias, el sistema evalúa la "huella digital" de rendimiento de cada piloto a través de 7 dimensiones clave basadas en su ELO histórico, consistencia y dominio sobre su compañero de equipo.
""")

st.divider()

# ── Tarjetas de navegación ────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="nav-card">
        <h3>🧬 ADN del Piloto</h3>
        <p>Buscá un piloto y temporada específica para ver su radar multidimensional, descubrir a qué arquetipo pertenece y encontrar sus "gemelos" históricos.</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="nav-card">
        <h3>⚔️ Duelo de Pilotos</h3>
        <p>Enfrentá a dos pilotos cara a cara. Analizá en qué métricas específicas se sacan ventaja y descubrí en qué año de la historia fueron más parecidos.</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="nav-card">
        <h3>🎛️ Piloto Hipotético</h3>
        <p>Convertite en jefe de equipo: definí tu perfil de piloto ideal moviendo los sliders (7D) y dejá que el modelo encuentre quién se acerca más en la realidad.</p>
    </div>
    """, unsafe_allow_html=True)

# ── Sección de Arquetipos y Métricas ──────────────────────────────────────────
st.divider()
col_arch, col_stats = st.columns([2, 1])

with col_arch:
    st.markdown("### 🧩 Los 5 Arquetipos Históricos")
    st.markdown("Nuestro algoritmo K-Means agrupó naturalmente la historia de la F1 en estos perfiles:")
    
    # Badges exactos con los colores y nombres del modelo de clustering
    st.markdown("""
    - <span class='badge' style='background-color:#E1060022; color:#E10600; border:1px solid #E10600'>El Dominador Absoluto</span> ELO altísimo, consistencia letal. No deja puntos sobre la mesa.
    - <span class='badge' style='background-color:#f39c1222; color:#f39c12; border:1px solid #f39c12'>El Maestro a Una Vuelta</span> Exprime el auto en Qualy, pero sufre o pierde ELO los domingos en carrera.
    - <span class='badge' style='background-color:#27ae6022; color:#27ae60; border:1px solid #27ae60'>El Muro Defensivo</span> Súper consistente. Poca volatilidad, suma siempre lo que el monoplaza le permite.
    - <span class='badge' style='background-color:#2980b922; color:#2980b9; border:1px solid #2980b9'>El Piloto Escudero</span> Fuerte diferencia negativa contra su compañero, pero cumple su rol de apoyo.
    - <span class='badge' style='background-color:#7f8c8d22; color:#7f8c8d; border:1px solid #7f8c8d'>El Sobreviviente</span> Fondo de parrilla. Lucha contra el monoplaza y la alta volatilidad.
    """, unsafe_allow_html=True)

with col_stats:
    st.markdown("### 📊 Estado de la Base de Datos")
    if stats:
        st.metric("Temporadas Analizadas", stats["periodo"])
        st.metric("Vectores (Piloto/Año)", f"{stats['vectores']} registros")
        st.metric("Pilotos Únicos", stats["pilotos"])
        st.metric("Dimensiones por Vector", f"{stats['dimensiones']} (Normalizadas)")
    else:
        st.warning("⚠️ No se encontró el archivo de datos. Asegúrate de ejecutar los pipelines de modelos primero.")