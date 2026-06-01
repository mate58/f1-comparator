import os
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client
import time

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def get_race_ids(season):
    """Obtiene de Supabase los IDs internos de las carreras de una temporada específica."""
    response = supabase.table('races').select('race_id, round').eq('season', season).execute()
    return {item['round']: item['race_id'] for item in response.data}

def fetch_and_upload_results():
    print("Iniciando descarga histórica de resultados por temporada...")
    
    for season in range(1950, 2027):
        print(f"\n--- Procesando Temporada {season} ---")
        
        race_mapping = get_race_ids(season)
        if not race_mapping:
            print(f"No hay carreras registradas para {season}. Saltando...")
            continue

        url = f"https://api.jolpi.ca/ergast/f1/{season}/results.json?limit=1000"
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error en API para el año {season}")
            continue
            
        races_data = response.json()['MRData']['RaceTable']['Races']
        
        # SOLUCIÓN: Usamos un diccionario en lugar de una lista para evitar duplicados históricos
        season_results_dict = {}
        
        for race in races_data:
            rnd = int(race['round'])
            
            if rnd not in race_mapping:
                continue
                
            race_db_id = race_mapping[rnd]
            
            for result in race.get('Results', []):
                fastest_lap = result.get('FastestLap', {})
                fastest_lap_time = fastest_lap.get('Time', {}).get('time', None)
                fastest_lap_speed = fastest_lap.get('AverageSpeed', {}).get('speed', None)
                fastest_lap_rank = fastest_lap.get('rank', None)
                
                try:
                    pos = int(result['position'])
                except ValueError:
                    pos = None

                try:
                    num = int(result['number'])
                except ValueError:
                    num = None

                # El ID único del resultado
                result_id = f"{season}_{rnd}_{result['Driver']['driverId']}"
                
                record = {
                    'result_id': result_id,
                    'race_id': race_db_id,
                    'driver_id': result['Driver']['driverId'],
                    'constructor_id': result['Constructor']['constructorId'],
                    'number': num,
                    'grid': int(result['grid']),
                    'position': pos,
                    'position_text': result['positionText'],
                    'points': float(result['points']),
                    'laps': int(result['laps']),
                    'time': result.get('Time', {}).get('time', None),
                    'status': result['status'],
                    'fastest_lap_rank': int(fastest_lap_rank) if fastest_lap_rank else None,
                    'fastest_lap_time': fastest_lap_time,
                    'fastest_lap_speed': fastest_lap_speed
                }
                
                # Solo lo agregamos si no existe previamente (ignora el auto compartido)
                if result_id not in season_results_dict:
                    season_results_dict[result_id] = record
                else:
                    print(f"  [!] Auto compartido histórico omitido: Piloto {result['Driver']['driverId']} en Ronda {rnd}")
        
        # Convertimos el diccionario limpio de nuevo a una lista para subirlo
        all_season_results = list(season_results_dict.values())
        
        if all_season_results:
            print(f"Subiendo {len(all_season_results)} resultados únicos del año {season}...")
            supabase.table('results').upsert(all_season_results).execute()
        
        time.sleep(0.5)
        
    print("\nDescarga histórica de resultados completada")


if __name__ == "__main__":
    fetch_and_upload_results()