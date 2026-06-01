import os
import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def fetch_and_upload_drivers():
    print("Descargando TODOS los pilotos históricos...")
    all_drivers = []
    offset = 0
    limit = 100
    
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/drivers.json?limit={limit}&offset={offset}"
        response = requests.get(url)
        
        if response.status_code == 200:
            drivers = response.json()['MRData']['DriverTable']['Drivers']
            if not drivers: # Si la lista está vacía, terminamos
                break
            all_drivers.extend(drivers)
            offset += limit
        else:
            print("Error conectando a la API")
            break

    df = pd.DataFrame(all_drivers)
    df_clean = df[['driverId', 'permanentNumber', 'code', 'givenName', 'familyName', 'dateOfBirth', 'nationality']].copy()
    df_clean.rename(columns={
        'driverId': 'driver_id',
        'permanentNumber': 'permanent_number',
        'givenName': 'given_name',
        'familyName': 'family_name',
        'dateOfBirth': 'date_of_birth'
    }, inplace=True)
    
    # Rellenamos nulos correctamente
    df_clean['permanent_number'] = df_clean['permanent_number'].fillna('')
    df_clean['code'] = df_clean['code'].fillna('')
    df_clean = df_clean.where(pd.notna(df_clean), None)
    
    records = df_clean.to_dict(orient='records')
    print(f"Subiendo {len(records)} pilotos a Supabase...")
    supabase.table('drivers').upsert(records).execute()
    print("✅ Pilotos cargados exitosamente.")

def fetch_and_upload_constructors():
    print("\nDescargando TODAS las escuderías históricas...")
    all_constructors = []
    offset = 0
    limit = 100
    
    while True:
        url = f"https://api.jolpi.ca/ergast/f1/constructors.json?limit={limit}&offset={offset}"
        response = requests.get(url)
        
        if response.status_code == 200:
            constructors = response.json()['MRData']['ConstructorTable']['Constructors']
            if not constructors:
                break
            all_constructors.extend(constructors)
            offset += limit
        else:
            break

    df = pd.DataFrame(all_constructors)
    df_clean = df[['constructorId', 'name', 'nationality']].copy()
    df_clean.rename(columns={'constructorId': 'constructor_id'}, inplace=True)
    
    records = df_clean.to_dict(orient='records')
    print(f"Subiendo {len(records)} escuderías a Supabase...")
    supabase.table('constructors').upsert(records).execute()
    print("✅ Escuderías cargadas exitosamente.")

if __name__ == "__main__":
    fetch_and_upload_drivers()
    fetch_and_upload_constructors()