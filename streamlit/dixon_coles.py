import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from typing import Dict, List, Optional, Any, Tuple

EPS = 1e-12

class DixonColesEloModel:
    def __init__(
        self,
        xi: float = 0.003,
        max_goals: int = 6,
        elo_scale: float = 300.0,
        reg_lambda: float = 0.01,
        param_clip_log_rate: float = 3.0,
        n_restarts: int = 3,
        random_state: int = 42,
    ) -> None:
        self.xi = float(xi)
        self.max_goals = int(max_goals)
        self.elo_scale = float(elo_scale)
        self.reg_lambda = float(reg_lambda)
        self.param_clip_log_rate = float(param_clip_log_rate)
        self.n_restarts = int(n_restarts)
        self.random_state = int(random_state)

        self.teams_: Optional[pd.Index] = None
        self.team_to_idx_: Optional[Dict[str, int]] = None
        self.params_: Optional[np.ndarray] = None
        self.optim_result_: Optional[Any] = None

    @staticmethod
    def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
        if x == 0 and y == 0:
            return 1.0 - (lam * mu * rho)
        if x == 0 and y == 1:
            return 1.0 + (lam * rho)
        if x == 1 and y == 0:
            return 1.0 + (mu * rho)
        if x == 1 and y == 1:
            return 1.0 - rho
        return 1.0

    def _unpack_params(self, params: np.ndarray, n_teams: int) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        attack = params[:n_teams].copy()
        defense = params[n_teams : 2 * n_teams].copy()
        home_adv = float(params[-3])
        rho = float(params[-2])
        beta_elo = float(params[-1])

        attack -= attack.mean()
        defense -= defense.mean()
        return attack, defense, home_adv, rho, beta_elo

    def _compute_rates(
        self,
        home_attack: float,
        home_defense: float,
        away_attack: float,
        away_defense: float,
        elo_diff: float,
        home_adv: float,
        beta_elo: float,
    ) -> Tuple[float, float]:
        elo_term = beta_elo * (elo_diff / self.elo_scale)
        log_lam = np.clip(home_adv + home_attack + away_defense + elo_term, -4.0, self.param_clip_log_rate)
        log_mu = np.clip(away_attack + home_defense - elo_term, -4.0, self.param_clip_log_rate)
        return float(np.exp(log_lam)), float(np.exp(log_mu))

    def _neg_log_likelihood(
        self,
        params: np.ndarray,
        h_idx: np.ndarray,
        a_idx: np.ndarray,
        h_goals: np.ndarray,
        a_goals: np.ndarray,
        elo_diff: np.ndarray,
        weights: np.ndarray,
        n_teams: int,
    ) -> float:
        attack, defense, home_adv, rho, beta_elo = self._unpack_params(params, n_teams)

        elo_term = beta_elo * (elo_diff / self.elo_scale)
        log_lam = np.clip(home_adv + attack[h_idx] + defense[a_idx] + elo_term, -4.0, self.param_clip_log_rate)
        log_mu = np.clip(attack[a_idx] + defense[h_idx] - elo_term, -4.0, self.param_clip_log_rate)
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        ll = weights * (poisson.logpmf(h_goals, lam) + poisson.logpmf(a_goals, mu))

        tau = np.ones_like(h_goals, dtype=float)
        mask_00 = (h_goals == 0) & (a_goals == 0)
        mask_01 = (h_goals == 0) & (a_goals == 1)
        mask_10 = (h_goals == 1) & (a_goals == 0)
        mask_11 = (h_goals == 1) & (a_goals == 1)

        tau[mask_00] = 1.0 - (lam[mask_00] * mu[mask_00] * rho)
        tau[mask_01] = 1.0 + (lam[mask_01] * rho)
        tau[mask_10] = 1.0 + (mu[mask_10] * rho)
        tau[mask_11] = 1.0 - rho

        tau = np.clip(tau, EPS, None)
        ll += weights * np.log(tau)

        reg = self.reg_lambda * (
            np.sum(attack ** 2)
            + np.sum(defense ** 2)
            + 0.25 * (home_adv ** 2)
            + 0.25 * (beta_elo ** 2)
            + 0.25 * (rho ** 2)
        )

        return float(-(np.sum(ll) - reg))

    def fit(self, matches: pd.DataFrame) -> "DixonColesEloModel":
        required = ["home_team_id", "away_team_id", "home_goals", "away_goals", "start_time", "elo_diff"]
        missing = [c for c in required if c not in matches.columns]
        if missing:
            raise ValueError(f"Faltan columnas para fit: {missing}")

        data = matches[required].copy()
        data["start_time"] = pd.to_datetime(data["start_time"], utc=True)

        teams = pd.Index(sorted(set(data["home_team_id"]).union(set(data["away_team_id"]))))
        self.teams_ = teams
        self.team_to_idx_ = {team_id: idx for idx, team_id in enumerate(teams)}
        n_teams = len(teams)

        h_idx = data["home_team_id"].map(self.team_to_idx_).values.astype(int)
        a_idx = data["away_team_id"].map(self.team_to_idx_).values.astype(int)

        h_goals = data["home_goals"].values.astype(int)
        a_goals = data["away_goals"].values.astype(int)
        elo_diff = data["elo_diff"].values.astype(float)

        reference_date = data["start_time"].max()
        days_ago = (reference_date - data["start_time"]).dt.days.values.astype(float)
        weights = np.exp(-self.xi * np.maximum(days_ago, 0.0))

        rng = np.random.default_rng(self.random_state)

        bounds = [(-3.0, 3.0)] * (2 * n_teams) + [(-1.0, 2.0), (-0.5, 0.5), (-2.0, 2.0)]

        best_result = None
        best_fun = float("inf")

        for _ in range(self.n_restarts):
            x0 = np.zeros(2 * n_teams + 3, dtype=float)
            x0[-3] = 0.20 + rng.normal(0.0, 0.05)   # home_adv
            x0[-2] = -0.05 + rng.normal(0.0, 0.02)  # rho
            x0[-1] = 0.10 + rng.normal(0.0, 0.05)   # beta_elo

            result = minimize(
                fun=self._neg_log_likelihood,
                x0=x0,
                args=(h_idx, a_idx, h_goals, a_goals, elo_diff, weights, n_teams),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 800},
            )

            if result.fun < best_fun:
                best_result = result
                best_fun = float(result.fun)

        if best_result is None:
            raise RuntimeError("La optimización falló en todos los reinicios.")

        self.params_ = best_result.x
        self.optim_result_ = best_result
        return self

    def predict_expected_goals(self, matches: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if self.params_ is None or self.teams_ is None or self.team_to_idx_ is None:
            raise RuntimeError("El modelo no está entrenado.")

        attack, defense, home_adv, _, beta_elo = self._unpack_params(self.params_, len(self.teams_))

        mean_attack = float(attack.mean())
        mean_defense = float(defense.mean())

        h_idx_raw = matches["home_team_id"].map(self.team_to_idx_).values
        a_idx_raw = matches["away_team_id"].map(self.team_to_idx_).values

        h_known = ~pd.isna(h_idx_raw)
        a_known = ~pd.isna(a_idx_raw)

        h_idx = np.zeros(len(matches), dtype=int)
        a_idx = np.zeros(len(matches), dtype=int)
        h_idx[h_known] = h_idx_raw[h_known].astype(int)
        a_idx[a_known] = a_idx_raw[a_known].astype(int)

        h_att_vals = np.full(len(matches), mean_attack)
        h_def_vals = np.full(len(matches), mean_defense)
        a_att_vals = np.full(len(matches), mean_attack)
        a_def_vals = np.full(len(matches), mean_defense)

        h_att_vals[h_known] = attack[h_idx[h_known]]
        h_def_vals[h_known] = defense[h_idx[h_known]]
        a_att_vals[a_known] = attack[a_idx[a_known]]
        a_def_vals[a_known] = defense[a_idx[a_known]]

        elo_term = beta_elo * (matches["elo_diff"].values.astype(float) / self.elo_scale)
        log_lam = np.clip(home_adv + h_att_vals + a_def_vals + elo_term, -4.0, self.param_clip_log_rate)
        log_mu = np.clip(a_att_vals + h_def_vals - elo_term, -4.0, self.param_clip_log_rate)

        return np.exp(log_lam), np.exp(log_mu)

    def score_matrix(
        self,
        home_team_id: str,
        away_team_id: str,
        elo_diff: float,
        max_goals: Optional[int] = None,
    ) -> Tuple[np.ndarray, float, float]:
        if self.params_ is None or self.teams_ is None or self.team_to_idx_ is None:
            raise RuntimeError("El modelo no está entrenado.")

        max_goals = self.max_goals if max_goals is None else int(max_goals)

        attack, defense, home_adv, rho, beta_elo = self._unpack_params(self.params_, len(self.teams_))

        mean_attack = float(attack.mean())
        mean_defense = float(defense.mean())

        h_idx = self.team_to_idx_.get(home_team_id)
        a_idx = self.team_to_idx_.get(away_team_id)

        h_att = float(attack[h_idx]) if h_idx is not None else mean_attack
        h_def = float(defense[h_idx]) if h_idx is not None else mean_defense
        a_att = float(attack[a_idx]) if a_idx is not None else mean_attack
        a_def = float(defense[a_idx]) if a_idx is not None else mean_defense

        lam, mu = self._compute_rates(
            home_attack=h_att,
            home_defense=h_def,
            away_attack=a_att,
            away_defense=a_def,
            elo_diff=float(elo_diff),
            home_adv=home_adv,
            beta_elo=beta_elo,
        )

        matrix = np.zeros((max_goals + 1, max_goals + 1), dtype=float)
        for x in range(max_goals + 1):
            for y in range(max_goals + 1):
                matrix[x, y] = self._tau(x, y, lam, mu, rho) * poisson.pmf(x, lam) * poisson.pmf(y, mu)

        matrix /= matrix.sum()
        return matrix, lam, mu
