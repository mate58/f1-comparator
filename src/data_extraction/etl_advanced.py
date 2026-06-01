import os
import pandas as pd
import fastf1
import warnings
from datetime import datetime
from supabase import create_client, Client

# --- 1. CREAMOS LA CARPETA TEMPORAL EN COLAB ---
os.makedirs('f1_cache', exist_ok=True)

# --- 2. ACTIVAMOS EL CACHÉ APUNTANDO A ESA CARPETA ---
fastf1.Cache.enable_cache('f1_cache')

# --- 3. CONEXIÓN A SUPABASE ---
# (Asegurate de haber ejecutado antes la celda con tus claves de entorno)
supabase_url = os.environ.get("https://xnrdoeaqghavfndknygy.supabase.co")
supabase_key = os.environ.get("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhucmRvZWFxZ2hhdmZuZGtueWd5Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3ODcwNTk4OSwiZXhwIjoyMDk0MjgxOTg5fQ.UGBA8YpA7N_pi6m32dtNuIozE8f97g48zNAvm01VCug")
supabase: Client = create_client(supabase_url, supabase_key)

# Ignorar advertencias molestas de pandas/fastf1 en la consola
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_race_ids(season):
    """Obtiene los IDs de Supabase para cruzar con FastF1"""
    response = supabase.table('races').select('race_id, round').eq('season', season).execute()
    return {item['round']: item['race_id'] for item in response.data}

def get_driver_ids():
    """Obtiene un diccionario de Código (VER) -> Driver_ID (verstappen)"""
    response = supabase.table('drivers').select('driver_id, code').execute()
    return {item['code']: item['driver_id'] for item in response.data if item['code']}

def extract_telemetry_2026():
    print("Iniciando motor de telemetría FastF1 (Carga Incremental 2026)...")
    driver_map = get_driver_ids()
    hoy = datetime.now() # Capturamos la fecha actual exacta
    
    # Rango fijado exclusivamente para la temporada en curso
    for season in range(2018, 2027):
        print(f"\n--- Descargando Telemetría: Temporada {season} ---")
        race_mapping = get_race_ids(season)
        
        # Obtener el calendario oficial de esa temporada
        schedule = fastf1.get_event_schedule(season)
        
        for _, event in schedule.iterrows():
            rnd = event['RoundNumber']
            fecha_carrera = event['EventDate']
            
            # --- LA BARRERA TEMPORAL INTELIGENTE ---
            # Si la fecha de la carrera es en el futuro, frenamos por completo
            if pd.notna(fecha_carrera) and fecha_carrera > hoy:
                print(f"\n ⏸️ La Ronda {rnd} ({event['EventName']}) es en el futuro.")
                print(" 🏁 Carga incremental finalizada exitosamente hasta la fecha de hoy.")
                break
            # ---------------------------------------

            # Saltamos los test de pretemporada (Round 0) o si no está en Supabase
            if rnd == 0 or rnd not in race_mapping:
                continue
                
            race_db_id = race_mapping[rnd]
            print(f" > Procesando Round {rnd}: {event['EventName']}")
            
            try:
                # Cargar la sesión de la carrera ('R' = Race)
                session = fastf1.get_session(season, rnd, 'R')
                session.load(telemetry=True, weather=True, messages=False)
                
                # Resumen del clima
                weather = session.weather_data
                air_temp = float(weather['AirTemp'].mean()) if not weather.empty else None
                track_temp = float(weather['TrackTemp'].mean()) if not weather.empty else None
                is_raining = bool((weather['Rainfall'] > 0).any()) if not weather.empty else False
                
                telemetry_records = []
                
                # Procesar cada piloto por su código de 3 letras (ej. VER, HAM)
                for code in session.results['Abbreviation']:
                    if pd.isna(code) or code not in driver_map:
                        continue
                        
                    driver_id = driver_map[code]
                    
                    try:
                        # Extraer vueltas válidas del piloto
                        laps = session.laps.pick_driver(code)
                        if laps.empty: continue
                        
                        # Extraer métricas clave
                        top_speed = float(laps['SpeedI1'].max()) if 'SpeedI1' in laps.columns else None
                        avg_tyre = float(laps['TyreLife'].mean()) if 'TyreLife' in laps.columns else None
                        
                        # Limpiar NaN de pandas a None de Python (para SQL)
                        top_speed = None if pd.isna(top_speed) else top_speed
                        avg_tyre = None if pd.isna(avg_tyre) else avg_tyre
                        
                        telemetry_records.append({
                            'telemetry_id': f"{season}_{rnd}_{driver_id}",
                            'race_id': race_db_id,
                            'driver_id': driver_id,
                            'code': code,
                            'top_speed_kmh': top_speed,
                            'avg_tyre_age': avg_tyre,
                            'air_temp_mean': air_temp,
                            'track_temp_mean': track_temp,
                            'is_raining': is_raining
                        })
                    except Exception as e:
                        print(f"    - Error procesando piloto {code}: {e}")
                
                # Subir el paquete de datos del GP a Supabase usando UPSERT (no duplica datos)
                if telemetry_records:
                    supabase.table('race_telemetry_summary').upsert(telemetry_records).execute()
                    
            except Exception as e:
                print(f" [!] Error al cargar la sesión (Round {rnd}): {e}")

if __name__ == "__main__":
    extract_telemetry_2026()