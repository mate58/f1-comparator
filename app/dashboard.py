import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from supabase import create_client, Client
from dotenv import load_dotenv
import random

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="F1 Oracle | Predictor de Carreras", layout="wide", page_icon="🏎️")
load_dotenv()

@st.cache_resource
def init_connection():
    return create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

supabase = init_connection()

@st.cache_resource
def load_model():
    # ¡CORRECCIÓN!: Ahora cargamos la V2 que generó tu train_model.py
    return joblib.load("src/models/saved_models/xgboost_f1_v2.pkl")

model = load_model()

# --- FUNCIONES DE BASE DE DATOS (DINÁMICAS) ---
@st.cache_data(ttl=3600)
def get_current_season_circuits(season=2026):
    try:
        response = supabase.table('races').select('race_id, race_name').eq('season', season).execute()
        if response.data:
            return {item['race_name']: item['race_id'] for item in response.data}
        return {"Gran Premio de Prueba (Sin datos)": 0}
    except Exception as e:
        st.error(f"Error conectando a la BD: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_active_grid(season=2026):
    """Obtiene dinámicamente los pilotos de la grilla más reciente con resultados reales"""
    try:
        # 1. Consultamos tu Vista Analítica ordenando por la fecha de carrera más reciente
        # Como la vista nace de un JOIN con 'results', ignora automáticamente el futuro
        view_response = supabase.table('v_analytical_base_table').select('driver_id').order('race_date', desc=True).limit(25).execute()
        
        if not view_response.data:
            raise Exception("No se encontraron registros con resultados en la vista analítica.")
            
        # 2. Extraemos los IDs únicos de los pilotos que corrieron esa última carrera
        active_driver_ids = list(set([row['driver_id'] for row in view_response.data]))
        
        # 3. Traemos los datos de la tabla drivers (id y code) para esos pilotos específicos
        drivers_response = supabase.table('drivers').select('driver_id, code').in_('driver_id', active_driver_ids).execute()
        
        if not drivers_response.data:
            raise Exception("No se pudieron emparejar los IDs con la tabla de pilotos.")
            
        return drivers_response.data
        
    except Exception as e:
        st.error(f"🚨 Error buscando la grilla actual: {e}")
        st.warning("Usando grilla de emergencia para desarrollo.")
        # PLAN B: Grilla de respaldo por si la base de datos está caída
        return [
            {'driver_id': 'verstappen', 'code': 'VER'}, {'driver_id': 'norris', 'code': 'NOR'},
            {'driver_id': 'leclerc', 'code': 'LEC'}, {'driver_id': 'sainz', 'code': 'SAI'},
            {'driver_id': 'hamilton', 'code': 'HAM'}, {'driver_id': 'russell', 'code': 'RUS'},
            {'driver_id': 'piastri', 'code': 'PIA'}, {'driver_id': 'perez', 'code': 'PER'},
            {'driver_id': 'alonso', 'code': 'ALO'}, {'driver_id': 'albon', 'code': 'ALB'},
            {'driver_id': 'stroll', 'code': 'STR'}, {'driver_id': 'tsunoda', 'code': 'TSU'},
            {'driver_id': 'ricciardo', 'code': 'RIC'}, {'driver_id': 'hulkenberg', 'code': 'HUL'},
            {'driver_id': 'magnussen', 'code': 'MAG'}, {'driver_id': 'gasly', 'code': 'GAS'},
            {'driver_id': 'ocon', 'code': 'OCO'}, {'driver_id': 'bottas', 'code': 'BOT'},
            {'driver_id': 'zhou', 'code': 'ZHO'}, {'driver_id': 'sargeant', 'code': 'SAR'}
        ]
    
# --- INTERFAZ LATERAL: PARÁMETROS DEL MODELO ---
st.sidebar.image("https://logodownload.org/wp-content/uploads/2016/11/formula-1-logo-7.png", width=150)
st.sidebar.header("⚙️ Motor de Predicción")
st.sidebar.write("Ajustá el entorno y la estrategia:")

st.sidebar.subheader("Condiciones de Pista")
is_raining = st.sidebar.toggle("🌦️ Simular Lluvia (Wet Race)", value=False)
track_temp = st.sidebar.slider("🌡️ Temperatura de Pista (°C)", min_value=15.0, max_value=50.0, value=30.0)

st.sidebar.subheader("Estrategia Global")
total_pit_stops = st.sidebar.slider("🛠️ Cantidad de Pit Stops", min_value=0, max_value=4, value=2)
avg_pit_duration_ms = st.sidebar.slider("⏱️ Tiempo en Pit Lane (ms)", min_value=20000, max_value=35000, value=25000)

# --- CUERPO PRINCIPAL ---
st.title("🏆 Simulador de Gran Premio")
st.markdown("Calcula las probabilidades de victoria utilizando XGBoost y datos de telemetría.")

# 1. Cargar datos dinámicos
circuits_dict = get_current_season_circuits(2026)
circuit_names = list(circuits_dict.keys())
active_drivers = get_active_grid(2026)

# 2. Selección del escenario
col1, col2 = st.columns(2)
with col1:
    selected_race_name = st.selectbox("Seleccionar Gran Premio:", circuit_names)
    selected_race_id = circuits_dict.get(selected_race_name)

with col2:
    grid_order = st.selectbox("Asumir posiciones de largada (Clasificación):", 
                              ["Orden aleatorio (Simulación)", 
                               "Última carrera conocida"])

st.divider()

# --- LÓGICA DE SIMULACIÓN REAL ---
if st.button("🏁 Iniciar Simulación de Carrera", type="primary", use_container_width=True):
    if not active_drivers:
        st.error("No se pudo cargar la grilla de pilotos desde la base de datos.")
    else:
        with st.spinner('Procesando datos y calculando probabilidades con XGBoost V2...'):
            
            predict_data = []
            driver_names = []
            
            # Mezclamos la grilla para simular una clasificación
            grid_positions = list(range(1, len(active_drivers) + 1))
            if grid_order == "Orden aleatorio (Simulación)":
                random.shuffle(grid_positions)
            
            for index, driver in enumerate(active_drivers):
                pos = grid_positions[index]
                
                # ¡LA CLAVE!: Acá pasamos EXACTAMENTE las 9 variables que pide tu train_model.py
                feature_row = {
                    'starting_position': pos, 
                    'driver_age': random.uniform(22.0, 39.0),  # Simulación de edad
                    'driver_form_3_races': random.uniform(0.0, 60.0), # Simulación de puntos recientes
                    'qualifying_position': pos, 
                    'total_pit_stops': total_pit_stops, 
                    'avg_pit_duration_ms': avg_pit_duration_ms, 
                    'top_speed_kmh': random.uniform(315.0, 340.0), # Simulación de velocidad
                    'track_temp_mean': track_temp, 
                    'is_raining': float(is_raining) # Pasamos a float como hiciste en tu limpieza
                }
                predict_data.append(feature_row)
                
                # Arreglo del nombre
                nombre_limpio = driver.get('driver_id', f"Piloto {index}").replace("_", " ").title()
                driver_names.append(nombre_limpio)
            
            # Convertimos a DataFrame
            df_to_predict = pd.DataFrame(predict_data)
            
            try:
                # LA PREDICCIÓN REAL
                predictions = model.predict(df_to_predict)
                
                results_df = pd.DataFrame({
                    'Piloto': driver_names,
                    'Score Predictivo': predictions
                })
                
                results_df = results_df.sort_values(by='Score Predictivo', ascending=True).reset_index(drop=True)
                
                # Normalización de probabilidades
                results_df['Probabilidad de Victoria'] = ((1 / results_df['Score Predictivo']) / (1 / results_df['Score Predictivo']).sum()) * 100
                results_df['Posición Estimada'] = range(1, len(results_df) + 1)
                
                st.success(f"Simulación completada para {selected_race_name}")
                
                podium = results_df.head(3)
                c1, c2, c3 = st.columns(3)
                c1.metric(label="🥇 1er Lugar", value=podium.iloc[0]['Piloto'], delta=f"{podium.iloc[0]['Probabilidad de Victoria']:.1f}% prob.")
                c2.metric(label="🥈 2do Lugar", value=podium.iloc[1]['Piloto'], delta=f"{podium.iloc[1]['Probabilidad de Victoria']:.1f}% prob.")
                c3.metric(label="🥉 3er Lugar", value=podium.iloc[2]['Piloto'], delta=f"{podium.iloc[2]['Probabilidad de Victoria']:.1f}% prob.")
                
                st.divider()
                st.markdown("### 📊 Parrilla de Resultados Completa")
                
                st.dataframe(
                    results_df[['Posición Estimada', 'Piloto', 'Probabilidad de Victoria']],
                    column_config={
                        "Probabilidad de Victoria": st.column_config.ProgressColumn(
                            "Prob. de Victoria",
                            help="Probabilidad matemática de ganar la carrera según la IA",
                            format="%.1f %%",
                            min_value=0,
                            max_value=100,
                        ),
                    },
                    hide_index=True,
                    use_container_width=True
                )

            except Exception as e:
                st.error("🚨 Error ejecutando el Modelo AI.")
                st.warning(f"Detalle técnico: {e}")