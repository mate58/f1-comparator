"""
dimensionality.py
=================
Reducción de dimensionalidad comparativa: UMAP vs PCA.
(Versión Backend: Solo calcula y guarda, sin ventanas emergentes).
"""

import pandas as pd
import umap.umap_ as umap
from sklearn.decomposition import PCA

FEATURE_COLS = [
    "elo_pico_temporada_norm",
    "elo_final_temporada_norm",
    "elo_volatilidad_norm",
    "elo_qualy_vs_race_delta_norm",
    "elo_delta_companero_norm",
    "win_rate_duelos_carrera_norm",
    "elo_trend_norm"
]

def aplicar_reduccion():
    print("🌌 Iniciando motores de Reducción de Dimensionalidad (Silencioso)...")
    
    try:
        df = pd.read_csv('driver_features_clustered.csv')
    except FileNotFoundError:
        print("❌ Error: No se encontró 'driver_features_clustered.csv'.")
        return

    # UMAP
    print("⚙️ Calculando UMAP...")
    reducer_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    embedding_umap = reducer_umap.fit_transform(df[FEATURE_COLS])
    df['umap_x'] = embedding_umap[:, 0]
    df['umap_y'] = embedding_umap[:, 1]

    # PCA
    print("⚙️ Calculando PCA...")
    reducer_pca = PCA(n_components=2, random_state=42)
    embedding_pca = reducer_pca.fit_transform(df[FEATURE_COLS])
    df['pca_x'] = embedding_pca[:, 0]
    df['pca_y'] = embedding_pca[:, 1]

    # EXPORTACIÓN
    df.to_csv('driver_features_final.csv', index=False)
    print(f"✅ ¡Archivo 'driver_features_final.csv' guardado con éxito!")

if __name__ == "__main__":
    aplicar_reduccion()