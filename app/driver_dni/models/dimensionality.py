"""
dimensionality.py
=================
Reducción de dimensionalidad con UMAP para el proyecto F1 Driver DNA.

Por qué UMAP y no PCA ni t-SNE
--------------------------------
La elección del algoritmo de reducción no es cosmética: define qué estructura
del espacio de features sobrevive en la proyección 2D que usamos para
clustering y visualización.

  PCA (Análisis de Componentes Principales)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  PCA busca las direcciones de máxima varianza asumiendo que las relaciones
  entre variables son LINEALES. Proyecta los datos sobre ejes ortogonales
  que maximizan la varianza explicada.

  Problema para nuestro caso: un piloto no es "más o menos" en cada feature
  de forma independiente. La interacción entre elo_delta_companero y
  car_bonus_avg es no lineal: ganarle al compañero con un auto malo vale
  mucho más que con el mejor auto de la grilla. PCA destruye esa estructura.

  t-SNE (t-distributed Stochastic Neighbor Embedding)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  t-SNE es excelente para visualización de clusters pero tiene tres problemas:
  1. No preserva estructura global: las distancias ENTRE clusters no son
     interpretables. Solo las distancias DENTRO de un cluster son confiables.
  2. No permite transformar nuevos puntos (requiere reentrenamiento).
  3. Es no determinístico y muy sensible a `perplexity`.

  UMAP (Uniform Manifold Approximation and Projection)
  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  UMAP modela los datos como una variedad topológica (manifold) y busca
  una proyección de menor dimensión que preserve tanto la estructura LOCAL
  (vecinos cercanos) como la GLOBAL (distancias entre grupos).

  Ventajas concretas para este proyecto:
  - Permite `transform()` sobre nuevos puntos → el piloto hipotético del
    dashboard puede proyectarse en el espacio ya entrenado.
  - Más rápido que t-SNE para datasets medianos (2000+ vectores).
  - `n_neighbors` es interpretable: cuántas carreras/temporadas de
    "contexto" considera al construir el manifold.
  - Con `metric='cosine'` es coherente con la similitud que usamos en
    similarity.py: el mismo concepto de "dirección del vector" guía
    tanto la reducción como la comparación.

Parámetros clave
----------------
  n_neighbors  : 15  → equilibrio entre estructura local y global.
                       Valores bajos (5) → más clusters fragmentados.
                       Valores altos (50) → estructura más suavizada.
  min_dist     : 0.1 → qué tan "apretados" están los puntos dentro
                       de un cluster. 0.0 = máxima compresión.
  n_components : 2   → para visualización. El pipeline interno usa 3
                       para clustering (más información preservada).
  metric       : 'cosine' → coherente con similarity.py. Dos pilotos
                       con perfiles proporcionales (mismo estilo, distinto
                       ELO absoluto por el auto) quedan cerca.
  random_state : 42  → reproducibilidad total del experimento.
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
    n_components = 3,   # Para clustering: más info preservada
    metric       = "cosine",
    random_state = 42,
)

FEATURE_COLS = [
    "elo_pico_temporada",
    "elo_final_temporada",
    "elo_volatilidad",
    "elo_qualy_vs_race_delta",
    "elo_delta_companero",
    "win_rate_duelos_carrera",
    "car_bonus_avg",
    "elo_trend",
    "remontada_rate",
]

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────
# ENTRENAMIENTO
# ──────────────────────────────────────────────

def entrenar_umap(df_features: pd.DataFrame) -> tuple:
    """
    Entrena dos reducers UMAP:
      - reducer_2d : para visualización en el scatter plot del dashboard
      - reducer_3d : para clustering (más información preservada en 3 ejes)

    Guarda ambos modelos en artifacts/ para uso posterior sin reentrenar.

    Parámetros
    ----------
    df_features : DataFrame normalizado con columnas FEATURE_COLS

    Retorna
    -------
    (X_2d, X_3d, reducer_2d, reducer_3d)
    """
    X = df_features[FEATURE_COLS].fillna(0).values
    print(f"Entrenando UMAP sobre {X.shape[0]} vectores de {X.shape[1]} dimensiones...")

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
    """
    Proyecta un piloto nuevo en el espacio UMAP ya entrenado, sin reentrenar.
    Esta capacidad es exclusiva de UMAP (vs. t-SNE, que no la soporta).
    """
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
    """
    Adjunta las coordenadas UMAP (2D y 3D) al DataFrame de features.
    Opcionalmente agrega las etiquetas de cluster.

    Retorna el DataFrame enriquecido listo para el dashboard.
    """
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
    df_features = pd.read_csv("driver_features.csv")
    X_2d, X_3d, reducer_2d, reducer_3d = entrenar_umap(df_features)

    df_proyectado = construir_df_proyectado(df_features, X_2d, X_3d)
    df_proyectado.to_csv("driver_features_umap.csv", index=False)
    print(f"\n✓ driver_features_umap.csv guardado ({len(df_proyectado)} registros)")