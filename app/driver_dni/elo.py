import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# 1. CONFIGURACIÓN ELO Y SUPABASE
# ==========================================
load_dotenv()
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Faltan las credenciales de Supabase en el .env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

ELO_BASE = 1000
K_ESTANDAR = 32
K_ACCIDENTE = 50
K_QUALY = 10
BONO_POR_POSICION_GANADA = 2.5 # Puntos extra por cada lugar avanzado

elo_pilotos = {}
historial_elo = []

# ==========================================
# 2. FUNCIONES DE EXTRACCIÓN Y PREPARACIÓN
# ==========================================
def fetch_all_supabase_data(table_name: str, batch_size: int = 1000) -> pd.DataFrame:
    """Extrae todos los registros de una tabla de Supabase paginando."""
    print(f"Descargando tabla: {table_name}...")
    all_data = []
    start = 0
    while True:
        end = start + batch_size - 1
        response = supabase.table(table_name).select("*").range(start, end).execute()
        data = response.data
        if not data:
            break
        all_data.extend(data)
        if len(data) < batch_size:
            break
        start += batch_size
    return pd.DataFrame(all_data)

def preparar_df_carreras():
    """Une las tablas de Supabase para formar el df_carreras que espera la lógica ELO."""
    df_results = fetch_all_supabase_data("results")
    df_qualifying = fetch_all_supabase_data("qualifying")
    df_constructors = fetch_all_supabase_data("constructor_championship")

    # 1. Tomamos results como base (ya contiene 'year', 'race_id', 'driver_id' y 'constructor_id')
    df_carreras = df_results.copy()

    # 2. Unir con clasificaciones para obtener la posición de qualy
    df_carreras = pd.merge(
        df_carreras, 
        df_qualifying[['race_id', 'driver_id', 'position']],
        on=['race_id', 'driver_id'], 
        how='left', 
        suffixes=('', '_qualy')
    )
    # Renombramos la position original a race_position y la nueva a qualy_position
    df_carreras.rename(columns={'position_qualy': 'qualy_position', 'position': 'race_position'}, inplace=True)

    # 3. Unir con constructores para obtener la posición del equipo (para el Car Bonus)
    df_carreras = pd.merge(
        df_carreras, 
        df_constructors[['year', 'constructor_id', 'position']],
        on=['year', 'constructor_id'],
        how='left'
    )
    # Renombramos la position del campeonato a position_const
    df_carreras.rename(columns={'position': 'position_const'}, inplace=True)

    # 4. Limpieza de tipos de datos para evitar errores matemáticos con nulos
    df_carreras['qualy_position'] = pd.to_numeric(df_carreras['qualy_position'], errors='coerce')
    df_carreras['race_position'] = pd.to_numeric(df_carreras['race_position'], errors='coerce')
    df_carreras['grid'] = pd.to_numeric(df_carreras['grid'], errors='coerce').fillna(0)
    
    # 5. Calcular Car Bonus
    df_carreras['const_pos'] = pd.to_numeric(df_carreras['position_const'], errors='coerce').fillna(5)
    df_carreras['car_bonus'] = (11 - df_carreras['const_pos'].clip(1, 10)) * 10

    # 6. Ordenar cronológicamente para que el ELO simule el paso del tiempo
    df_carreras.sort_values(by=['year', 'race_id'], inplace=True)
    
    return df_carreras

# ==========================================
# 3. LÓGICA ELO (TU CÓDIGO EXACTO)
# ==========================================
def categorizar_status(status):
    status_str = str(status).lower()
    if status_str in ['accident', 'collision', 'spun off']:
        return 'accidente'
    elif 'finished' in status_str or 'lap' in status_str:
        return 'termino'
    else:
        return 'mecanica'

def calcular_probabilidad(elo_a, bono_a, elo_b, bono_b):
    re_a = elo_a + bono_a
    re_b = elo_b + bono_b
    return 1 / (1 + 10 ** ((re_b - re_a) / 400))

def procesar_elo():
    print("\nIniciando preparación de datos...")
    df_carreras = preparar_df_carreras()
    
    print("\nProcesando el motor ELO...")
    for (year, race_id), carrera_data in df_carreras.groupby(['year', 'race_id']):
        
        pilotos_evento = carrera_data.to_dict('records')
        cambios_qualy = {p['driver_id']: 0 for p in pilotos_evento}
        cambios_race = {p['driver_id']: 0 for p in pilotos_evento}
        registro_carrera = {}

        # ==========================================
        # 1. INICIALIZACIÓN SEGURA PARA TODOS LOS PILOTOS
        # ==========================================
        for p in pilotos_evento:
            d_id = p['driver_id']
            if d_id not in elo_pilotos:
                elo_pilotos[d_id] = ELO_BASE
            registro_carrera[d_id] = {}
        
        # ==========================================
        # FASE 1: PROCESAMIENTO DE CLASIFICACIÓN
        # ==========================================
        pilotos_qualy = [p for p in pilotos_evento if pd.notnull(p['qualy_position'])]
        
        for i in range(len(pilotos_qualy)):
            p_A = pilotos_qualy[i]
            d_id_A = p_A['driver_id']
            elo_A = elo_pilotos[d_id_A]
            
            if 'elo_qualy_inicial' not in registro_carrera[d_id_A]:
                registro_carrera[d_id_A]['elo_qualy_inicial'] = elo_A
                
            for j in range(i + 1, len(pilotos_qualy)):
                p_B = pilotos_qualy[j]
                d_id_B = p_B['driver_id']
                elo_B = elo_pilotos[d_id_B]
                
                prob_A = calcular_probabilidad(elo_A, p_A['car_bonus'], elo_B, p_B['car_bonus'])
                prob_B = 1 - prob_A
                
                resultado_A = 1 if p_A['qualy_position'] < p_B['qualy_position'] else 0
                resultado_B = 1 - resultado_A
                
                cambios_qualy[d_id_A] += K_QUALY * (resultado_A - prob_A)
                cambios_qualy[d_id_B] += K_QUALY * (resultado_B - prob_B)

        num_rivales_qualy = max(1, len(pilotos_qualy) - 1)
        for p in pilotos_qualy:
            d_id = p['driver_id']
            elo_pilotos[d_id] += (cambios_qualy[d_id] / num_rivales_qualy)
            registro_carrera[d_id]['elo_qualy_final'] = elo_pilotos[d_id]

        # ==========================================
        # FASE 2: PROCESAMIENTO DE CARRERA
        # ==========================================
        pilotos_carrera = [p for p in pilotos_evento if pd.notnull(p['race_position']) and categorizar_status(p['status']) != 'mecanica']
        total_participantes = len(pilotos_evento)
        
        for i in range(len(pilotos_carrera)):
            p_A = pilotos_carrera[i]
            driver_id_A = p_A['driver_id']
            elo_A = elo_pilotos[driver_id_A]
            registro_carrera[driver_id_A]['elo_race_inicial'] = elo_A
            
            categoria_A = categorizar_status(p_A['status'])
            K_A = K_ACCIDENTE if categoria_A == 'accidente' else K_ESTANDAR
                
            for j in range(i + 1, len(pilotos_carrera)):
                p_B = pilotos_carrera[j]
                driver_id_B = p_B['driver_id']
                elo_B = elo_pilotos[driver_id_B]
                
                categoria_B = categorizar_status(p_B['status'])
                K_B = K_ACCIDENTE if categoria_B == 'accidente' else K_ESTANDAR
                
                prob_A = calcular_probabilidad(elo_A, p_A['car_bonus'], elo_B, p_B['car_bonus'])
                prob_B = 1 - prob_A
                
                resultado_A = 1 if p_A['race_position'] < p_B['race_position'] else 0
                resultado_B = 1 - resultado_A
                
                cambios_race[driver_id_A] += K_A * (resultado_A - prob_A)
                cambios_race[driver_id_B] += K_B * (resultado_B - prob_B)

        num_rivales_carrera = max(1, len(pilotos_carrera) - 1)
        for p in pilotos_carrera:
            driver_id = p['driver_id']
            
            grid_real = total_participantes if p['grid'] == 0 else p['grid']
            posiciones_ganadas = grid_real - p['race_position']
            
            elo_duelos = (cambios_race[driver_id] / num_rivales_carrera)
            bono_remontada = posiciones_ganadas * BONO_POR_POSICION_GANADA
            
            elo_pilotos[driver_id] += (elo_duelos + bono_remontada)
            registro_carrera[driver_id]['elo_race_final'] = elo_pilotos[driver_id]

        # ==========================================
        # FASE 3: COMPILACIÓN DEL HISTORIAL
        # ==========================================
        for p in pilotos_evento:
            driver_id = p['driver_id']
            
            elo_q_i = registro_carrera[driver_id].get('elo_qualy_inicial', elo_pilotos[driver_id])
            elo_q_f = registro_carrera[driver_id].get('elo_qualy_final', elo_q_i)
            elo_r_i = registro_carrera[driver_id].get('elo_race_inicial', elo_q_f)
            elo_r_f = registro_carrera[driver_id].get('elo_race_final', elo_r_i)
            
            elo_pilotos[driver_id] = elo_r_f

            historial_elo.append({
                'id': f"{driver_id}_{race_id}",
                'race_id': race_id,
                'piloto_id': driver_id,
                'year': year,
                'elo_qualy_inicial': round(elo_q_i),
                'elo_qualy_final': round(elo_q_f),
                'elo_race_inicial': round(elo_r_i),
                'elo_race_final': round(elo_r_f)
            })

    # ==========================================
    # 4. EXPORTACIÓN FINAL
    # ==========================================
    df_final = pd.DataFrame(historial_elo)
    df_final.to_csv('driver_elo_para_supabase.csv', index=False)
    print(f"\n¡Archivo local generado! Total registros: {len(df_final)}")

    # 🚨 FIX: Quitamos la columna 'id' que sobra, pero mantenemos todas las métricas iniciales y finales
    if 'id' in df_final.columns:
        df_supabase = df_final.drop(columns=['id'])
    else:
        df_supabase = df_final

    print(f"🚀 Subiendo {len(df_supabase)} registros a la tabla 'driver_elo' en Supabase...")
    records = df_supabase.to_dict("records")
    
    batch_size = 500
    for i in range(0, len(records), batch_size):
        batch = records[i:i+batch_size]
        try:
            supabase.table("driver_elo").upsert(batch).execute()
            print(f"  → Subidos {min(i+batch_size, len(records))}/{len(records)}...", end="\r")
        except Exception as e:
            print(f"\n❌ Error en el lote {i}: {e}")

    print("\n✅ ¡Motor ELO procesado y subido a la base de datos con éxito!")

if __name__ == "__main__":
    procesar_elo()