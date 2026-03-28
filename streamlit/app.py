import math
import sys
from typing import Dict, Tuple, List

# Compatibility patch for Python 3.13+ where imghdr was removed
try:
    import imghdr
except ImportError:
    from types import ModuleType
    mock_imghdr = ModuleType("imghdr")
    mock_imghdr.what = lambda file, h=None: None
    sys.modules["imghdr"] = mock_imghdr

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import poisson

import sys
from dixon_coles import DixonColesEloModel
# Shim to allow joblib to find the class if it was pickled as __main__.DixonColesEloModel
import __main__
__main__.DixonColesEloModel = DixonColesEloModel


# ============================================================
# Config
# ============================================================
st.set_page_config(page_title="Predictor LaLiga", page_icon="⚽", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #eee;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #e63946;
        color: white;
        font-weight: bold;
    }
    .stSelectbox label {
        color: #1d3557;
        font-weight: bold;
    }
    h1, h2, h3 {
        color: #1d3557;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DIXON_MODEL_PATH = BASE_DIR / "models" / "dixon_coles_model.pkl"
POISSON_MODEL_PATH = BASE_DIR / "models" / "poisson_model.pkl"
TEAM_CATALOG_PATH = BASE_DIR / "dataset" /"team_catalog.csv"


# ============================================================
# Helpers
# ============================================================
def build_team_catalog_from_ratings(ratings_obj) -> pd.DataFrame:
    """
    Fallback simple team catalog if no CSV exists.
    Names will be the team_id itself.
    """
    if isinstance(ratings_obj, dict):
        team_ids = sorted(ratings_obj.keys())
    elif isinstance(ratings_obj, pd.DataFrame) and "team_id" in ratings_obj.columns:
        team_ids = sorted(ratings_obj["team_id"].astype(str).unique().tolist())
    else:
        team_ids = []

    return pd.DataFrame({"team_id": team_ids, "team_name": team_ids})


def ensure_ratings_dict(ratings_obj) -> Dict[str, float]:
    if isinstance(ratings_obj, dict):
        return {str(k): float(v) for k, v in ratings_obj.items()}

    if isinstance(ratings_obj, pd.DataFrame):
        if "team_id" not in ratings_obj.columns:
            raise ValueError("final_ratings DataFrame must contain a 'team_id' column.")

        value_col = None
        for candidate in ["final_elo", "elo", "rating"]:
            if candidate in ratings_obj.columns:
                value_col = candidate
                break

        if value_col is None:
            raise ValueError("Could not find an Elo/rating column in final_ratings DataFrame.")

        return dict(zip(ratings_obj["team_id"].astype(str), ratings_obj[value_col].astype(float)))

    raise ValueError("Unsupported ratings object format.")


def compute_fixture_probabilities(matrix: np.ndarray) -> Dict[str, float]:
    p_home = 0.0
    p_draw = 0.0
    p_away = 0.0

    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            p = float(matrix[i, j])
            if i > j:
                p_home += p
            elif i == j:
                p_draw += p
            else:
                p_away += p

    return {
        "home_win": p_home,
        "draw": p_draw,
        "away_win": p_away,
    }


def top_scorelines(matrix: np.ndarray, top_n: int = 5) -> List[Dict[str, float]]:
    flat = []
    rows, cols = matrix.shape
    for i in range(rows):
        for j in range(cols):
            flat.append({
                "home_goals": i,
                "away_goals": j,
                "probability": float(matrix[i, j]),
            })

    flat = sorted(flat, key=lambda x: x["probability"], reverse=True)
    return flat[:top_n]


def predict_fixture(model, current_ratings: Dict[str, float], home_team_id: str, away_team_id: str, max_goals: int = 6):
    home_rating = float(current_ratings.get(home_team_id, 1500.0))
    away_rating = float(current_ratings.get(away_team_id, 1500.0))
    elo_diff = home_rating - away_rating

    matrix, lam, mu = model.score_matrix(
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        elo_diff=elo_diff,
        max_goals=max_goals,
    )

    probs = compute_fixture_probabilities(matrix)

    return {
        "home_team_id": home_team_id,
        "away_team_id": away_team_id,
        "elo_diff": elo_diff,
        "home_expected_goals": float(lam),
        "away_expected_goals": float(mu),
        "probabilities": probs,
        "top_scorelines": top_scorelines(matrix, top_n=5),
    }


def xg_to_match_probs(home_xg: float, away_xg: float, max_goals: int = 10) -> Dict[str, float]:
    probs = np.zeros(3)

    for i in range(max_goals + 1):
        p_i = poisson.pmf(i, home_xg)
        for j in range(max_goals + 1):
            p = p_i * poisson.pmf(j, away_xg)
            if i > j:
                probs[0] += p
            elif i == j:
                probs[1] += p
            else:
                probs[2] += p

    probs = probs / probs.sum()
    return {
        "home_win": float(probs[0]),
        "draw": float(probs[1]),
        "away_win": float(probs[2]),
    }


def build_poisson_features(home_team_id: str, away_team_id: str, columns: List[str]) -> pd.DataFrame:
    X = pd.DataFrame([
        {
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
        }
    ])

    X = pd.get_dummies(X, columns=["home_team_id", "away_team_id"], drop_first=False)
    X = X.reindex(columns=columns, fill_value=0)
    return X


def format_pct(x: float) -> str:
    return f"{x * 100:.1f}%"


# ============================================================
# Load models and data
# ============================================================
@st.cache_resource

def load_assets():
    dixon_data = joblib.load(DIXON_MODEL_PATH)
    poisson_data = joblib.load(POISSON_MODEL_PATH)

    dixon_model = dixon_data["model"]
    ratings = ensure_ratings_dict(dixon_data["ratings"])

    poisson_home = poisson_data["home_model"]
    poisson_away = poisson_data["away_model"]
    poisson_columns = poisson_data["columns"]

     # Load or build team catalog
    try:
        catalog_df = pd.read_csv(TEAM_CATALOG_PATH)
        catalog_df["team_id"] = catalog_df["team_id"].astype(str)
        catalog_df["team_name"] = catalog_df["team_name"].astype(str)
    except Exception:
        catalog_df = pd.DataFrame(columns=["team_id", "team_name"])

    # IDs from model
    model_team_ids = list(ratings.keys())
    model_df = pd.DataFrame({"team_id": model_team_ids})
    model_df["team_id"] = model_df["team_id"].astype(str)

    # Merge to get names, fallback to ID if missing in CSV
    team_catalog = pd.merge(model_df, catalog_df, on="team_id", how="left")
    team_catalog["team_name"] = team_catalog["team_name"].fillna(team_catalog["team_id"])
    team_catalog = team_catalog.drop_duplicates(subset=["team_id"]).sort_values("team_name").reset_index(drop=True)

    return dixon_model, ratings, poisson_home, poisson_away, poisson_columns, team_catalog


try:
    (
        dixon_model,
        dixon_ratings,
        poisson_home_model,
        poisson_away_model,
        poisson_columns,
        team_catalog,
    ) = load_assets()
except Exception as e:
    st.error("No se pudieron cargar los modelos.")
    st.exception(e)
    st.stop()


# ============================================================
# UI
# ============================================================
st.title("⚽ LaLiga Match Predictor")
st.caption("Predicción de goles esperados y probabilidades 1X2 con Dixon-Coles + Elo y Poisson.")

team_name_to_id = dict(zip(team_catalog["team_name"], team_catalog["team_id"]))
team_names = team_catalog["team_name"].tolist()

if len(team_names) < 2:
    st.error("No hay suficientes equipos en el catálogo para hacer predicciones.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    home_team_name = st.selectbox("Equipo local", team_names, index=0)
with col2:
    default_away_index = 1 if len(team_names) > 1 else 0
    away_team_name = st.selectbox("Equipo visitante", team_names, index=default_away_index)

max_goals = 6
show_scorelines = st.checkbox("Mostrar marcadores más probables", value=True)

if home_team_name == away_team_name:
    st.warning("Selecciona dos equipos diferentes.")
    st.stop()

home_team_id = team_name_to_id[home_team_name]
away_team_id = team_name_to_id[away_team_name]

if st.button("Analizar Partido", type="primary"):
    with st.spinner('Procesando estadísticas y simulando partido...'):
        # Dixon-Coles + Elo
        dixon_pred = predict_fixture(
            model=dixon_model,
            current_ratings=dixon_ratings,
            home_team_id=home_team_id,
            away_team_id=away_team_id,
            max_goals=max_goals,
        )

        # Poisson
        X_single = build_poisson_features(home_team_id, away_team_id, poisson_columns)
        poisson_home_xg = float(poisson_home_model.predict(X_single)[0])
        poisson_away_xg = float(poisson_away_model.predict(X_single)[0])
        poisson_probs = xg_to_match_probs(poisson_home_xg, poisson_away_xg)

    st.markdown(f"## 🏟️ {home_team_name} vs {away_team_name}")
    
    # 1. Elo Info
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.metric("Elo Local", f"{dixon_ratings.get(home_team_id, 1500.0):.0f}")
    with col_e2:
        st.metric("Elo Visitante", f"{dixon_ratings.get(away_team_id, 1500.0):.0f}")
    with col_e3:
        diff = dixon_pred['elo_diff']
        st.metric("Diferencia Elo", f"{diff:+.0f}", delta=f"{diff:.0f}")

    st.markdown("---")

    # 2. Results Comparison
    st.subheader("📊 Probabilidades Victoria/Empate (1X2)")
    
    comparison = pd.DataFrame([
        {
            "Modelo": "Dixon-Coles + Elo",
            "Goles Exp. Local": round(dixon_pred["home_expected_goals"], 2),
            "Goles Exp. Visitante": round(dixon_pred["away_expected_goals"], 2),
            "Victoria Local": format_pct(dixon_pred["probabilities"]["home_win"]),
            "Empate": format_pct(dixon_pred["probabilities"]["draw"]),
            "Victoria Visitante": format_pct(dixon_pred["probabilities"]["away_win"]),
        },
        {
            "Modelo": "Regresión Poisson",
            "Goles Exp. Local": round(poisson_home_xg, 2),
            "Goles Exp. Visitante": round(poisson_away_xg, 2),
            "Victoria Local": format_pct(poisson_probs["home_win"]),
            "Empate": format_pct(poisson_probs["draw"]),
            "Victoria Visitante": format_pct(poisson_probs["away_win"]),
        },
    ])
    st.table(comparison)

    st.markdown("---")

    # 3. Model Details
    left, right = st.columns(2)

    with left:
        st.markdown("### 🎲 Dixon-Coles + Elo")
        st.info("Este modelo ajusta la probabilidad Poisson basándose en la correlación histórica de marcadores bajos y la fuerza relativa (Elo).")
        
        probs_df = pd.DataFrame([
            {"Resultado": f"Gana {home_team_name}", "Probabilidad": format_pct(dixon_pred["probabilities"]["home_win"]), "p": dixon_pred["probabilities"]["home_win"]},
            {"Resultado": "Empate", "Probabilidad": format_pct(dixon_pred["probabilities"]["draw"]), "p": dixon_pred["probabilities"]["draw"]},
            {"Resultado": f"Gana {away_team_name}", "Probabilidad": format_pct(dixon_pred["probabilities"]["away_win"]), "p": dixon_pred["probabilities"]["away_win"]},
        ]).sort_values("p", ascending=False).drop(columns=["p"])

        st.dataframe(probs_df)

        if show_scorelines:
            st.markdown("**🎯 Marcadores Exactos más probables**")
            scores = pd.DataFrame(dixon_pred["top_scorelines"])
            scores.columns = ["Goles Local", "Goles Visitante", "Probabilidad"]
            scores["Probabilidad"] = scores["Probabilidad"].apply(format_pct)
            st.dataframe(scores)

    with right:
        st.markdown("### 📈 Regresión Poisson")
        st.info("Modelo basado en ataques y defensas individuales por equipo mediante regresión generalizada.")
        
        probs_df_p = pd.DataFrame([
            {"Resultado": f"Gana {home_team_name}", "Probabilidad": format_pct(poisson_probs["home_win"]), "p": poisson_probs["home_win"]},
            {"Resultado": "Empate", "Probabilidad": format_pct(poisson_probs["draw"]), "p": poisson_probs["draw"]},
            {"Resultado": f"Gana {away_team_name}", "Probabilidad": format_pct(poisson_probs["away_win"]), "p": poisson_probs["away_win"]},
        ]).sort_values("p", ascending=False).drop(columns=["p"])

        st.dataframe(probs_df_p)
        st.metric("Promedio Goles Esperados", f"{(poisson_home_xg + poisson_away_xg):.2f}")

st.markdown("---")
st.caption(
    "Nota: Estas predicciones son basadas en modelos matemáticos y datos históricos. No garantizan resultados. "
    "Se recomienda el uso de `dixon_coles_model.pkl` y `poisson_model.pkl` actualizados."
)
