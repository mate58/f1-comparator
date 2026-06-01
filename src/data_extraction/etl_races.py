import os
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def fetch_and_upload_races():
    print("Descargando todas las carreras de la historia...")
    all_races = []
    offset = 0
    limit = 100
    
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/races.json?limit={limit}&offset={offset}"
        response = requests.get(url)
        
        if response.status_code == 200:
            races = response.json()['MRData']['RaceTable']['Races']
            if not races: 
                break
            all_races.extend(races)
            offset += limit
        else:
            print("Error en la API")
            break
            
    df = pd.DataFrame(all_races)
    
    # Limpieza
    df_clean = df[['season', 'round', 'raceName', 'date', 'Circuit']].copy()
    df_clean['circuit_id'] = df_clean['Circuit'].apply(lambda x: x['circuitId'])
    df_clean.drop(columns=['Circuit'], inplace=True)
    df_clean.rename(columns={'raceName': 'race_name'}, inplace=True)
    
    # Asegurar tipos numéricos
    df_clean['season'] = df_clean['season'].astype(int)
    df_clean['round'] = df_clean['round'].astype(int)
    
    records = df_clean.to_dict(orient='records')
    print(f"Subiendo {len(records)} carreras a Supabase...")
    
    # Al estar vacía la tabla, insertamos directamente
    supabase.table('races').insert(records).execute()
    print("Carreras históricas cargadas exitosamente.")

if __name__ == "__main__":
    fetch_and_upload_races()