"""
etl_constructors.py
===================
Descarga la lista de constructores (escuderías) de Jolpi API y los sube a Supabase.
"""

import os
import time
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def etl_constructors():
    print("⏳ Descargando Constructores de Jolpi API...")
    url = "http://api.jolpi.ca/ergast/f1/constructors.json?limit=1000"
    
    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"❌ Error al conectar con Jolpi: {response.status_code}")
            return
        
        data = response.json()
        constructors = data["MRData"]["ConstructorTable"]["Constructors"]
        
        # Procesamiento
        df = pd.DataFrame(constructors)
        df = df.rename(columns={"constructorId": "constructor_id"})
        
        # Limpieza agresiva de NaN para evitar el error de JSON
        df = df.replace({np.nan: None})
        
        records = df.to_dict("records")
        
        # Carga a Supabase
        print(f"🚀 Subiendo {len(records)} constructores a Supabase...")
        supabase.table("constructors").upsert(records).execute()
        
        print("✅ ¡Constructores cargados exitosamente!")
        
    except Exception as e:
        print(f"❌ Error en el proceso: {e}")

if __name__ == "__main__":
    etl_constructors()