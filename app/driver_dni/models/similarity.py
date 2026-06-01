"""
similarity.py
=============
Motor de similitud para el proyecto F1 Driver DNA.

Métricas implementadas
-----------------------
  1. Similitud Coseno (principal)
  2. K-Nearest Neighbors con métrica coseno (top-N similares)
  3. Similitud por dimensión (para el radar chart del dashboard)

Por qué similitud coseno y no distancia euclidiana
---------------------------------------------------
La distancia euclidiana mide cuánto difieren dos vectores en magnitud.
El problema: dos pilotos pueden tener perfiles idénticos de rendimiento
pero ELOs absolutamente distintos porque uno corrió con un auto dominante
y el otro con un auto de midfield.

Ejemplo concreto:
  Alonso 2005 (Renault campeón)  → ELO 1180, car_bonus=90
  Alonso 2012 (Ferrari luchando) → ELO 1095, car_bonus=40

La magnitud de sus vectores es muy diferente, pero la DIRECCIÓN
(proporción entre features) es similar: alto win_rate vs. compañero,
alta remontada, alta consistencia.

La similitud coseno mide exactamente esa dirección:
  cos(θ) = (A · B) / (|A| × |B|)

Resultado: 0.93 de similitud entre Alonso 2005 y Alonso 2012, capturando
que son el "mismo estilo de piloto" independientemente del auto.

Con distancia euclidiana hubieran quedado lejos por diferencia de ELO
absoluto, que es función del auto, no del piloto.

Rango de similitud coseno: [-1, 1]
  1.0  → vectores idénticos (mismo estilo)
  0.0  → vectores ortogonales (estilos sin relación)
 -1.0  → vectores opuestos (raro con features positivos)

Convertimos a [0, 100%] multiplicando por 100 para el dashboard.

KNN con métrica coseno
-----------------------
Para encontrar los top-N pilotos más similares usamos NearestNeighbors
de scikit-learn con metric='cosine'. Equivale a buscar los vectores con
mayor similitud coseno de forma eficiente (kd-tree o ball-tree interno).
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

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

# Etiquetas legibles para el radar chart del dashboard
FEATURE_LABELS = {
    "elo_pico_temporada"     : "ELO pico",
    "elo_final_temporada"    : "ELO final",
    "elo_volatilidad"        : "Consistencia",     # invertido en radar: baja volatilidad = alta consistencia
    "elo_qualy_vs_race_delta": "Domingo > Sábado",
    "elo_delta_companero"    : "vs. Compañero",
    "win_rate_duelos_carrera": "Win rate duelos",
    "car_bonus_avg"          : "Calidad auto",
    "elo_trend"              : "Progresión",
    "remontada_rate"         : "Remontadas",
}


# ──────────────────────────────────────────────
# 1. ENTRENAMIENTO DEL KNN
# ──────────────────────────────────────────────

def entrenar_knn(
    df_features: pd.DataFrame,
    n_neighbors: int = 11  # +1 porque el propio piloto es el vecino 0
) -> NearestNeighbors:
    """
    Entrena y persiste el modelo KNN sobre la matriz de features normalizada.

    n_neighbors=11 → retornará los 10 más similares (excluyendo al propio piloto).
    algorithm='brute' es óptimo para datasets pequeños (<10k vectores) con
    métrica coseno, ya que kd-tree no soporta métricas no euclidianas.
    """
    X = df_features[FEATURE_COLS].fillna(0).values

    knn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric="cosine",
        algorithm="brute"
    )
    knn.fit(X)
    joblib.dump(knn, ARTIFACTS_DIR / "knn.pkl")
    print(f"✓ KNN entrenado sobre {X.shape[0]} vectores → artifacts/knn.pkl")
    return knn


# ──────────────────────────────────────────────
# 2. TOP-N SIMILARES A UN PILOTO DADO
# ──────────────────────────────────────────────

def top_similares(
    piloto_id: str,
    year: int,
    df_features: pd.DataFrame,
    knn: NearestNeighbors,
    n: int = 5,
    excluir_mismo_piloto: bool = True
) -> pd.DataFrame:
    """
    Dado un piloto y temporada, retorna los N pilotos/temporadas más similares.

    Parámetros
    ----------
    piloto_id             : ID del piloto objetivo
    year                  : Temporada del piloto objetivo
    df_features           : DataFrame con features normalizados
    knn                   : Modelo KNN ya entrenado
    n                     : Cuántos similares retornar
    excluir_mismo_piloto  : Si True, excluye otras temporadas del mismo piloto

    Retorna
    -------
    DataFrame con columnas: piloto_id, year, similitud_pct, archetype
    """
    mask = (
        (df_features["piloto_id"].astype(str) == str(piloto_id)) &
        (df_features["year"] == year)
    )
    if not mask.any():
        raise ValueError(f"No se encontró {piloto_id} en {year}")

    idx_objetivo = df_features[mask].index[0]
    X = df_features[FEATURE_COLS].fillna(0).values
    vector = X[idx_objetivo].reshape(1, -1)

    distancias, indices = knn.kneighbors(vector, n_neighbors=n + 10)
    # distancia coseno ∈ [0, 1] → similitud = 1 - distancia
    distancias = distancias[0]
    indices    = indices[0]

    resultados = []
    for dist, idx in zip(distancias, indices):
        fila = df_features.iloc[idx]
        if excluir_mismo_piloto and str(fila["piloto_id"]) == str(piloto_id):
            continue
        similitud_pct = round((1 - dist) * 100, 1)
        resultados.append({
            "piloto_id"    : fila["piloto_id"],
            "year"         : fila["year"],
            "similitud_pct": similitud_pct,
            "archetype"    : fila.get("archetype", "—"),
        })
        if len(resultados) == n:
            break

    return pd.DataFrame(resultados)


# ──────────────────────────────────────────────
# 3. COMPARACIÓN DIRECTA ENTRE DOS PILOTOS
# ──────────────────────────────────────────────

def comparar_dos_pilotos(
    piloto_a: str, year_a: int,
    piloto_b: str, year_b: int,
    df_features: pd.DataFrame
) -> dict:
    """
    Calcula la similitud global y por dimensión entre dos pilotos/temporadas.

    Retorna un dict con:
      - similitud_global_pct : similitud coseno total (0-100%)
      - delta_por_feature    : diferencia por feature (para saber quién "gana" en cada eje)
      - radar_data           : datos para el radar chart superpuesto
    """
    def get_vector(pid, yr):
        mask = (
            (df_features["piloto_id"].astype(str) == str(pid)) &
            (df_features["year"] == yr)
        )
        if not mask.any():
            raise ValueError(f"No se encontró {pid} en {yr}")
        return df_features[mask][FEATURE_COLS].fillna(0).values[0]

    vec_a = get_vector(piloto_a, year_a)
    vec_b = get_vector(piloto_b, year_b)

    similitud_global = float(cosine_similarity([vec_a], [vec_b])[0][0])
    similitud_pct    = round(similitud_global * 100, 1)

    # Diferencia por feature: positivo → A supera a B en esa dimensión
    delta = {feat: round(float(va - vb), 3) for feat, va, vb in zip(FEATURE_COLS, vec_a, vec_b)}

    # Invertir volatilidad para que el radar sea "más alto = mejor"
    vec_a_radar = vec_a.copy()
    vec_b_radar = vec_b.copy()
    vol_idx = FEATURE_COLS.index("elo_volatilidad")
    vec_a_radar[vol_idx] = 1 - vec_a_radar[vol_idx]
    vec_b_radar[vol_idx] = 1 - vec_b_radar[vol_idx]

    radar_data = {
        "features"      : [FEATURE_LABELS[f] for f in FEATURE_COLS],
        "values_a"      : vec_a_radar.tolist(),
        "values_b"      : vec_b_radar.tolist(),
        "label_a"       : f"{piloto_a} ({year_a})",
        "label_b"       : f"{piloto_b} ({year_b})",
    }

    return {
        "similitud_global_pct": similitud_pct,
        "delta_por_feature"   : delta,
        "radar_data"          : radar_data,
    }


# ──────────────────────────────────────────────
# 4. PILOTO HIPOTÉTICO → MATCH HISTÓRICO
# ──────────────────────────────────────────────

def match_piloto_hipotetico(
    vector_custom: dict,
    df_features: pd.DataFrame,
    scaler,
    knn: NearestNeighbors,
    n: int = 3
) -> pd.DataFrame:
    """
    Dado un piloto hipotético definido por el usuario (vía sliders en el
    dashboard), encuentra los N pilotos históricos más similares.

    Parámetros
    ----------
    vector_custom : dict con valores en escala REAL (no normalizados)
                    Ejemplo: {"elo_pico_temporada": 1200, "car_bonus_avg": 20, ...}
    df_features   : DataFrame con features normalizados
    scaler        : MinMaxScaler entrenado en build_features.py
    knn           : Modelo KNN entrenado

    Retorna
    -------
    DataFrame con top-N pilotos históricos más parecidos al perfil ingresado.
    """
    # Construir y normalizar el vector custom
    vector_raw = np.array([[vector_custom.get(f, 0) for f in FEATURE_COLS]])
    vector_norm = scaler.transform(vector_raw)

    distancias, indices = knn.kneighbors(vector_norm, n_neighbors=n)
    distancias = distancias[0]
    indices    = indices[0]

    resultados = []
    for dist, idx in zip(distancias, indices):
        fila = df_features.iloc[idx]
        resultados.append({
            "piloto_id"    : fila["piloto_id"],
            "year"         : fila["year"],
            "similitud_pct": round((1 - dist) * 100, 1),
            "archetype"    : fila.get("archetype", "—"),
        })

    return pd.DataFrame(resultados)


# ──────────────────────────────────────────────
# 5. VISUALIZACIONES
# ──────────────────────────────────────────────

def graficar_radar(radar_data: dict) -> go.Figure:
    """
    Radar chart superpuesto para comparar dos pilotos.
    Preparado para st.plotly_chart() en Streamlit.
    """
    features = radar_data["features"]
    # Cerrar el polígono repitiendo el primer valor
    va = radar_data["values_a"] + [radar_data["values_a"][0]]
    vb = radar_data["values_b"] + [radar_data["values_b"][0]]
    cats = features + [features[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=va, theta=cats, fill="toself",
        name=radar_data["label_a"],
        fillcolor="rgba(225,6,0,0.15)",
        line=dict(color="#E10600", width=2)
    ))
    fig.add_trace(go.Scatterpolar(
        r=vb, theta=cats, fill="toself",
        name=radar_data["label_b"],
        fillcolor="rgba(0,144,208,0.15)",
        line=dict(color="#0090D0", width=2)
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        template="plotly_white",
        height=450
    )
    return fig


def graficar_barra_similares(df_similares: pd.DataFrame, titulo: str = "Top similares") -> go.Figure:
    """
    Barra horizontal de similitud para el ADN del piloto.
    Colores por arquetipo (si disponible).
    """
    ARCHETYPE_COLORS = {
        "Dominador absoluto"           : "#E10600",
        "Sobre-rendidor con mal auto"  : "#FF8700",
        "Clasificador puro"            : "#0090D0",
        "Consistente de midfield"      : "#00D2BE",
        "Piloto de desarrollo"         : "#9B0A4B",
        "Piloto irregular"             : "#888888",
    }
    colores = df_similares["archetype"].map(
        lambda a: ARCHETYPE_COLORS.get(a, "#AAAAAA")
    ).tolist()

    etiquetas = [
        f"{r['piloto_id']} ({r['year']})" for _, r in df_similares.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x=df_similares["similitud_pct"],
        y=etiquetas,
        orientation="h",
        marker_color=colores,
        text=df_similares["similitud_pct"].apply(lambda v: f"{v}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title=titulo,
        xaxis=dict(title="Similitud (%)", range=[0, 105]),
        yaxis=dict(autorange="reversed"),
        template="plotly_white",
        height=350
    )
    return fig


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    df = pd.read_csv("driver_features_clustered.csv")
    scaler = joblib.load("scaler.pkl")

    knn = entrenar_knn(df)

    # Ejemplo: ¿A quién se parece Alonso en 2005?
    print("\n=== Top 5 similares a alonso_2005 ===")
    similares = top_similares("alonso", 2005, df, knn)
    print(similares.to_string())

    # Ejemplo: Duelo Hamilton 2019 vs. Schumacher 2004
    print("\n=== Duelo Hamilton 2019 vs Michael Schumacher 2004 ===")
    # 👇 Aquí está el cambio clave: "michael_schumacher"
    resultado = comparar_dos_pilotos("hamilton", 2019, "michael_schumacher", 2004, df) 
    print(f"Similitud global: {resultado['similitud_global_pct']}%")