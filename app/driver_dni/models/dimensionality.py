"""
dimensionality.py
=================
Reducción de dimensionalidad con UMAP para el proyecto F1 Driver DNA.
"""

import joblib
import numpy as np
import pandas as pd
import umap
from pathlib import Path

# ──────────────────────────────────────────────
# CONFIGURACIÓN
# ──────────────────────────────────────────────

UMAP_2D_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 2,
    metric       = "cosine",
    random_state = 42,
)

UMAP_3D_PARAMS = dict(
    n_neighbors  = 15,
    min_dist     = 0.1,
    n_components = 3,   
    metric       = "cosine",
    random_state = 42,
)

# Actualizado a las 7 dimensiones reales (sin las variables de ruido)
FEATURE_COLS = [
    "elo_pico_temporada",
    "elo_final_temporada",
    "elo_volatilidad",
    "elo_qualy_vs_race_delta",
    "elo_delta_companero",
    "win_rate_duelos_carrera",
    "elo_trend",
]

# UMAP DEBE entrenarse sobre las columnas normalizadas
COLS_NORM = [f"{c}_norm" for c in FEATURE_COLS]

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────

def entrenar_umap(df_features: pd.DataFrame) -> tuple:
    """
    Entrena dos reducers UMAP usando las dimensiones NORMALIZADAS.
    """
    # Usamos COLS_NORM en lugar de FEATURE_COLS
    X = df_features[COLS_NORM].fillna(0).values
    print(f"Entrenando UMAP sobre {X.shape[0]} vectores de {X.shape[1]} dimensiones (normalizadas)...")

    # 2D — visualización
    reducer_2d = umap.UMAP(**UMAP_2D_PARAMS)
    X_2d = reducer_2d.fit_transform(X)
    print(f"  → 2D: {X_2d.shape}")

    # 3D — clustering
    reducer_3d = umap.UMAP(**UMAP_3D_PARAMS)
    X_3d = reducer_3d.fit_transform(X)
    print(f"  → 3D: {X_3d.shape}")

    # Persistencia
    joblib.dump(reducer_2d, ARTIFACTS_DIR / "umap_2d.pkl")
    joblib.dump(reducer_3d, ARTIFACTS_DIR / "umap_3d.pkl")
    print("  → Modelos guardados en artifacts/")

    return X_2d, X_3d, reducer_2d, reducer_3d


def proyectar_nuevo_piloto(
    vector_normalizado: np.ndarray,
    reducer_path: str | None = None
) -> np.ndarray:
    if reducer_path is None:
        reducer_path = str(Path(__file__).parent / "artifacts" / "umap_2d.pkl")
    reducer = joblib.load(reducer_path)
    return reducer.transform(vector_normalizado.reshape(1, -1))


# ──────────────────────────────────────────────
# CONSTRUCCIÓN DEL DATAFRAME FINAL CON COORDS
# ──────────────────────────────────────────────

def construir_df_proyectado(
    df_features: pd.DataFrame,
    X_2d: np.ndarray,
    X_3d: np.ndarray,
    labels_cluster: np.ndarray | None = None
) -> pd.DataFrame:
    df = df_features.copy()
    df["umap_x"]  = X_2d[:, 0]
    df["umap_y"]  = X_2d[:, 1]
    df["umap_x3"] = X_3d[:, 0]
    df["umap_y3"] = X_3d[:, 1]
    df["umap_z3"] = X_3d[:, 2]

    if labels_cluster is not None:
        df["cluster"] = labels_cluster

    return df


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    HERE = Path(__file__).parent
    PARENT = HERE.parent
    
    csv_path = PARENT / "driver_features.csv"
    
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo en {csv_path}. Ejecuta build_features.py primero.")
        
    df_features = pd.read_csv(csv_path)
    
    # Validamos que estén las columnas normalizadas
    if not all(col in df_features.columns for col in COLS_NORM):
        raise ValueError("El CSV no tiene las columnas _norm. Vuelve a ejecutar build_features.py.")

    X_2d, X_3d, reducer_2d, reducer_3d = entrenar_umap(df_features)

    df_proyectado = construir_df_proyectado(df_features, X_2d, X_3d)
    
    output_path = PARENT / "driver_features_umap.csv"
    df_proyectado.to_csv(output_path, index=False)
    print(f"\n✓ {output_path.name} guardado con éxito ({len(df_proyectado)} registros)")