"""RiskLens hackathon differentiators: contagion, reverse stress, Indian frictions,
and hypothetical headline-driven tail-risk scenarios.

Important: outputs from this module are analytical scenario estimates, not live
exchange signals, tax advice, regulatory certification, or guaranteed forecasts.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize
from scipy.stats import t as student_t


def _validated_returns(returns_df: pd.DataFrame, *, min_rows: int = 5) -> pd.DataFrame:
    if not isinstance(returns_df, pd.DataFrame) or returns_df.empty:
        raise ValueError("returns_df must be a non-empty DataFrame.")
    if returns_df.columns.has_duplicates:
        raise ValueError("Asset/ticker columns must be unique.")
    clean = returns_df.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna(how="any")
    if clean.shape[0] < min_rows:
        raise ValueError(f"At least {min_rows} complete observations are required.")
    if clean.shape[1] < 1:
        raise ValueError("At least one asset column is required.")
    if not np.isfinite(clean.to_numpy(dtype=float)).all():
        raise ValueError("Returns contain non-finite values after validation.")
    return clean.astype(float)


def _normalized_weights(weights: np.ndarray, n_assets: int) -> np.ndarray:
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != n_assets:
        raise ValueError(f"Expected {n_assets} portfolio weights, received {len(w)}.")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite and non-negative for this long-only scenario.")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("Portfolio weights must have a positive sum.")
    return w / total


class ContagionNetwork3DEngine:
    """Render a 3D dependence network and propagate a hypothetical downside shock.

    Edge weights are asymptotic lower-tail dependence estimates under a fitted-shape
    Student-t copula approximation; they are not causal contagion probabilities.
    """

    @staticmethod
    def compute_tail_dependence_matrix(returns_df: pd.DataFrame, nu: float = 5.0) -> np.ndarray:
        data = _validated_returns(returns_df)
        if not np.isfinite(nu) or nu <= 2:
            raise ValueError("Student-t degrees of freedom nu must be greater than 2.")
        tau = data.corr(method="kendall").to_numpy(dtype=float)
        tau = np.nan_to_num(tau, nan=0.0, posinf=0.0, neginf=0.0)
        rho = np.sin(0.5 * np.pi * np.clip(tau, -0.99, 0.99))
        np.fill_diagonal(rho, 1.0)
        arg = -np.sqrt(np.maximum((nu + 1.0) * (1.0 - rho) / np.maximum(1.0 + rho, 1e-8), 0.0))
        lower_tail = 2.0 * student_t.cdf(arg, df=nu + 1.0)
        lower_tail = np.clip(np.nan_to_num(lower_tail, nan=0.0), 0.0, 1.0)
        np.fill_diagonal(lower_tail, 0.0)
        return lower_tail

    @classmethod
    def simulate_epicenter_cascade(
        cls,
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        epicenter_ticker: str,
        initial_shock_pct: float = -0.12,
        nu: float = 5.0,
        edge_threshold: float = 0.18,
        damping: float = 0.45,
        max_abs_asset_shock: float = 0.95,
    ) -> dict[str, Any]:
        data = _validated_returns(returns_df)
        tickers = list(data.columns)
        n = len(tickers)
        if not np.isfinite(initial_shock_pct) or not -1.0 < initial_shock_pct < 0.0:
            raise ValueError("initial_shock_pct must be a negative return greater than -100%.")
        if not 0.0 <= edge_threshold <= 1.0:
            raise ValueError("edge_threshold must be between 0 and 1.")
        if not 0.0 <= damping <= 1.0:
            raise ValueError("damping must be between 0 and 1.")
        w = _normalized_weights(weights, n)
        lam_mat = cls.compute_tail_dependence_matrix(data, nu=nu)
        vol_vec = data.std(ddof=1).to_numpy() * np.sqrt(252.0)
        positive_vols = vol_vec[np.isfinite(vol_vec) & (vol_vec > 0)]
        typical_vol = float(positive_vols.mean()) if positive_vols.size else 1.0
        rel_vol = vol_vec / max(typical_vol, 1e-8)
        epi_idx = tickers.index(epicenter_ticker) if epicenter_ticker in tickers else 0

        hop0 = np.zeros(n, dtype=float)
        hop0[epi_idx] = initial_shock_pct
        hop1 = hop0.copy()
        for j in range(n):
            if j != epi_idx and lam_mat[epi_idx, j] >= edge_threshold:
                hop1[j] = initial_shock_pct * lam_mat[epi_idx, j] * np.clip(rel_vol[j], 0.6, 1.6)

        def propagate(previous: np.ndarray, direct_floor: np.ndarray | None = None) -> np.ndarray:
            next_shock = previous.copy()
            for j in range(n):
                if j == epi_idx:
                    continue
                incoming = sum(
                    previous[k] * lam_mat[k, j] * damping
                    for k in range(n)
                    if k != j and k != epi_idx and previous[k] < 0.0 and lam_mat[k, j] >= edge_threshold
                )
                next_shock[j] = np.clip(previous[j] + incoming, -max_abs_asset_shock, 0.0)
            if direct_floor is not None:
                next_shock = np.minimum(next_shock, direct_floor)
            return next_shock

        hop2 = propagate(hop1)
        hop3 = propagate(hop2)
        final_shock = np.clip(hop3, -max_abs_asset_shock, 0.0)

        # Deterministic spherical layout used for display only, not an inferred physical graph layout.
        angles_theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
        angles_phi = np.arccos(1.0 - 2.0 * (np.arange(n) + 0.5) / n)
        radius = 1.0 + 1.5 * np.abs(final_shock / max(abs(initial_shock_pct), 1e-4))
        coords = np.column_stack([
            radius * np.sin(angles_phi) * np.cos(angles_theta),
            radius * np.sin(angles_phi) * np.sin(angles_theta),
            radius * np.cos(angles_phi),
        ])
        return {
            "tickers": tickers,
            "weights": w,
            "lambda_matrix": lam_mat,
            "coords": coords,
            "hop0_shock": hop0,
            "hop1_shock": hop1,
            "hop2_shock": hop2,
            "hop3_final_shock": final_shock,
            "hop2_final_shock": final_shock,  # compatibility alias for prior prototype UI
            "portfolio_cascade_loss_pct": float(np.dot(w, final_shock)),
            "epicenter_ticker": tickers[epi_idx],
            "edge_threshold": float(edge_threshold),
            "damping": float(damping),
            "interpretation": "Hypothetical dependence-based cascade; not a causal or calibrated forecast.",
        }

    @staticmethod
    def build_3d_plotly_figure(cascade_data: dict[str, Any]) -> go.Figure:
        tickers = cascade_data["tickers"]
        coords = np.asarray(cascade_data["coords"], dtype=float)
        lam_mat = np.asarray(cascade_data["lambda_matrix"], dtype=float)
        shocks = np.asarray(cascade_data["hop3_final_shock"], dtype=float)
        weights = np.asarray(cascade_data["weights"], dtype=float)
        thresh = float(cascade_data["edge_threshold"])
        epi = cascade_data["epicenter_ticker"]
        edge_x: list[float | None] = []
        edge_y: list[float | None] = []
        edge_z: list[float | None] = []
        n = len(tickers)
        for i in range(n):
            for j in range(i + 1, n):
                if lam_mat[i, j] >= thresh:
                    edge_x.extend([coords[i, 0], coords[j, 0], None])
                    edge_y.extend([coords[i, 1], coords[j, 1], None])
                    edge_z.extend([coords[i, 2], coords[j, 2], None])
        edge_trace = go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z, mode="lines",
            line=dict(color="#8AA08C", width=3), opacity=0.45,
            hoverinfo="skip", name="Student-t lower-tail dependence edge",
        )
        hover_texts = [
            f"<b>{ticker}</b> {'(EPICENTER)' if ticker == epi else ''}<br>"
            f"Portfolio weight: {weights[i] * 100:.1f}%<br>"
            f"Hypothetical cascade shock: {shocks[i] * 100:.2f}%<br>"
            f"Max tail-dependence link: {float(np.max(lam_mat[i])):.2f}"
            for i, ticker in enumerate(tickers)
        ]
        node_trace = go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2], mode="markers+text",
            text=tickers, textposition="top center", textfont=dict(color="#28332C", size=11),
            hovertext=hover_texts, hoverinfo="text",
            marker=dict(size=np.clip(weights * 130.0, 12.0, 38.0), color=shocks * 100.0,
                        colorscale="YlGn", reversescale=True, colorbar=dict(title=dict(text="Shock (%)", font=dict(color="#28332C")), tickfont=dict(color="#5F6D61")),
                        line=dict(color="#F8F8F0", width=2)),
            name="Portfolio assets",
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            template="plotly_white", paper_bgcolor="#FFFEFA", plot_bgcolor="#FFFEFA", font=dict(color="#28332C", family="Aptos, Segoe UI, sans-serif"),
            margin=dict(l=0, r=0, b=0, t=40),
            title=dict(text=f"Tail-dependence network · scenario epicenter: {epi}", font=dict(color="#28332C", size=15)),
            scene=dict(bgcolor="#FFFEFA", xaxis=dict(showbackground=False, showticklabels=False, title=""),
                       yaxis=dict(showbackground=False, showticklabels=False, title=""),
                       zaxis=dict(showbackground=False, showticklabels=False, title="")),
            height=540,
        )
        return fig


class ReverseStressTestSolver:
    """Find the minimum Mahalanobis-distance return vector reaching a loss target."""

    @staticmethod
    def solve_most_probable_breaking_shock(
        returns_df: pd.DataFrame,
        weights: np.ndarray,
        portfolio_value: float,
        target_loss_currency: float,
        lower_return_bound: float = -0.65,
        upper_return_bound: float = 0.15,
    ) -> dict[str, Any]:
        data = _validated_returns(returns_df)
        tickers = list(data.columns)
        n = len(tickers)
        w = _normalized_weights(weights, n)
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive and finite.")
        if not np.isfinite(target_loss_currency) or target_loss_currency <= 0:
            raise ValueError("target_loss_currency must be positive and finite.")
        if not lower_return_bound < 0 < upper_return_bound or lower_return_bound <= -1 or upper_return_bound >= 1:
            raise ValueError("Return bounds must straddle zero and lie strictly between -1 and 1.")
        target_loss_pct = float(target_loss_currency / portfolio_value)
        if target_loss_pct > -lower_return_bound + 1e-10:
            return {
                "feasible": False, "solver_status": "infeasible_target",
                "message": "Target loss exceeds the maximum loss permitted by the per-asset return bounds.",
                "target_loss_currency": float(target_loss_currency), "target_loss_pct": target_loss_pct,
                "asset_shock_vector": [],
            }
        cov = np.asarray(data.cov().values, dtype=float)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        eigvals, eigvecs = np.linalg.eigh((cov + cov.T) / 2.0)
        cov_reg = eigvecs @ np.diag(np.maximum(eigvals, 1e-8)) @ eigvecs.T
        inv_cov = np.linalg.pinv(cov_reg)
        denom = float(w @ cov_reg @ w)
        analytic = -(target_loss_pct / max(denom, 1e-12)) * (cov_reg @ w)
        lower = np.full(n, lower_return_bound, dtype=float)
        upper = np.full(n, upper_return_bound, dtype=float)
        if np.all(analytic >= lower - 1e-8) and np.all(analytic <= upper + 1e-8) and abs(-w @ analytic - target_loss_pct) < 1e-6:
            r_star = analytic
            status = "analytical_solution"
            success = True
        else:
            # A constant shock is a feasible start whenever the target is within the bounds.
            x0 = np.full(n, -target_loss_pct, dtype=float)
            x0 = np.clip(x0, lower, upper)
            res = minimize(
                lambda r: float(r @ inv_cov @ r), x0, method="SLSQP",
                bounds=list(zip(lower, upper)),
                constraints=[{"type": "eq", "fun": lambda r: float(-w @ r - target_loss_pct)}],
                options={"maxiter": 500, "ftol": 1e-11},
            )
            success = bool(res.success and np.isfinite(res.x).all() and abs(-w @ res.x - target_loss_pct) < 1e-5)
            if not success:
                return {
                    "feasible": False, "solver_status": "optimizer_failed_or_infeasible",
                    "message": str(getattr(res, "message", "Optimization did not converge.")),
                    "target_loss_currency": float(target_loss_currency), "target_loss_pct": target_loss_pct,
                    "asset_shock_vector": [],
                }
            r_star = np.asarray(res.x, dtype=float)
            status = "SLSQP_constrained_solution"

        maha_sq = float(max(r_star @ inv_cov @ r_star, 0.0))
        maha_dist = float(np.sqrt(maha_sq))
        actual_loss_pct = float(-w @ r_star)
        breakdown: list[dict[str, Any]] = []
        for i, ticker in enumerate(tickers):
            asset_val = portfolio_value * w[i]
            asset_loss = -asset_val * r_star[i]
            sigma_i = float(np.sqrt(max(cov_reg[i, i], 0.0)))
            breakdown.append({
                "ticker": ticker, "weight_pct": round(w[i] * 100.0, 2),
                "required_return_shock_pct": round(r_star[i] * 100.0, 2),
                "daily_sigma_multiple": round(float(r_star[i] / max(sigma_i, 1e-8)), 2),
                "loss_contribution_currency": round(float(asset_loss), 2),
                "share_of_target_loss_pct": round(float(asset_loss / max(target_loss_currency, 1.0) * 100.0), 2),
            })
        return {
            "feasible": True, "solver_status": status,
            "target_loss_currency": float(target_loss_currency), "target_loss_pct": target_loss_pct,
            "actual_loss_pct": actual_loss_pct, "constraint_error_pct_points": (actual_loss_pct - target_loss_pct) * 100.0,
            "mahalanobis_distance_sigma": round(maha_dist, 4),
            "statistical_plausibility_index": round(float(np.exp(-0.35 * maha_dist) * 100.0), 2),
            "plausibility_index_note": "Uncalibrated heuristic display score, not a probability.",
            "covariance_regularization_eigenvalue_floor": 1e-8,
            "asset_shock_vector": breakdown,
        }


class IndianRegulatoryFrictionEngine:
    """Estimate cash-equity rebalance costs and a simple liquidity add-on to VaR.

    Rates are transparent configurable assumptions. Results omit several personal,
    broker-specific, and transaction-specific tax/charge details; not tax advice.
    """

    @staticmethod
    def compute_rebalancing_tax_and_stt(
        portfolio_value: float,
        current_weights: np.ndarray,
        target_weights: np.ndarray,
        unrealized_gain_pct_vec: np.ndarray | None = None,
        holding_days_vec: np.ndarray | None = None,
        *,
        stt_delivery_rate: float = 0.001,
        stamp_duty_buy_rate: float = 0.00015,
        exchange_and_sebi_fee_rate: float = 0.0000345,
        gst_rate: float = 0.18,
        stcg_tax_rate: float = 0.20,
        ltcg_tax_rate: float = 0.125,
        annual_ltcg_exemption_currency: float = 125_000.0,
        ltcg_gain_already_used_currency: float = 0.0,
        long_term_holding_days: int = 365,
    ) -> dict[str, Any]:
        if not np.isfinite(portfolio_value) or portfolio_value <= 0:
            raise ValueError("portfolio_value must be positive and finite.")
        current = np.asarray(current_weights, dtype=float).reshape(-1)
        target = np.asarray(target_weights, dtype=float).reshape(-1)
        if len(current) == 0 or len(current) != len(target):
            raise ValueError("current_weights and target_weights must have the same non-zero length.")
        for name, w in (("current_weights", current), ("target_weights", target)):
            if not np.isfinite(w).all() or np.any(w < 0) or not np.isclose(w.sum(), 1.0, atol=1e-4):
                raise ValueError(f"{name} must be finite, non-negative, and sum to 1.")
        n = len(current)
        gains = np.zeros(n) if unrealized_gain_pct_vec is None else np.asarray(unrealized_gain_pct_vec, dtype=float).reshape(-1)
        holding_days = np.zeros(n, dtype=int) if holding_days_vec is None else np.asarray(holding_days_vec, dtype=int).reshape(-1)
        if len(gains) != n or len(holding_days) != n or not np.isfinite(gains).all():
            raise ValueError("Gain and holding-period vectors must match the portfolio length and contain finite gains.")
        if np.any(gains <= -1) or np.any(holding_days < 0):
            raise ValueError("Gain percentages must exceed -100% and holding days must be non-negative.")
        rates = [stt_delivery_rate, stamp_duty_buy_rate, exchange_and_sebi_fee_rate, gst_rate,
                 stcg_tax_rate, ltcg_tax_rate, annual_ltcg_exemption_currency, ltcg_gain_already_used_currency]
        if not np.isfinite(rates).all() or any(x < 0 for x in rates):
            raise ValueError("Rates and exemption amounts must be finite and non-negative.")
        if int(long_term_holding_days) < 1:
            raise ValueError("long_term_holding_days must be positive.")

        diff = target - current
        buys_val = float(np.maximum(diff, 0.0).sum() * portfolio_value)
        sells_val = float(np.maximum(-diff, 0.0).sum() * portfolio_value)
        turnover = buys_val + sells_val
        stt = stt_delivery_rate * turnover  # delivery cash equity: buyer and seller legs
        stamp = stamp_duty_buy_rate * buys_val
        exchange_and_sebi = exchange_and_sebi_fee_rate * turnover
        gst = gst_rate * exchange_and_sebi

        stcg_gain = 0.0
        ltcg_gain = 0.0
        for i in range(n):
            sell_amount = max(-diff[i], 0.0) * portfolio_value
            if sell_amount <= 0 or gains[i] <= 0:
                continue
            # Input is gain as a return on acquisition cost; infer gain as a fraction of sale value.
            realized_gain = sell_amount * (gains[i] / (1.0 + gains[i]))
            if holding_days[i] < long_term_holding_days:
                stcg_gain += realized_gain
            else:
                ltcg_gain += realized_gain
        stcg_tax = stcg_tax_rate * stcg_gain
        # The 112A threshold is aggregate across eligible gains in the relevant tax year,
        # not a per-holding allowance. Prior usage is an explicit user assumption.
        ltcg_taxable_gain = max(ltcg_gain - max(annual_ltcg_exemption_currency - ltcg_gain_already_used_currency, 0.0), 0.0)
        ltcg_tax = ltcg_tax_rate * ltcg_taxable_gain
        total_friction = stt + stamp + exchange_and_sebi + gst + stcg_tax + ltcg_tax
        return {
            "buys_value_currency": round(buys_val, 2), "sells_value_currency": round(sells_val, 2),
            "total_turnover_currency": round(turnover, 2), "stt_currency": round(stt, 2),
            "stamp_duty_currency": round(stamp, 2), "exchange_and_sebi_fee_currency": round(exchange_and_sebi, 2),
            "gst_on_exchange_and_sebi_currency": round(gst, 2),
            "stamp_and_sebi_currency": round(stamp + exchange_and_sebi + gst, 2),
            "stcg_realized_gain_estimate_currency": round(stcg_gain, 2),
            "stcg_tax_20pct_currency": round(stcg_tax, 2),
            "ltcg_realized_gain_estimate_currency": round(ltcg_gain, 2),
            "ltcg_taxable_gain_estimate_currency": round(ltcg_taxable_gain, 2),
            "ltcg_tax_12_5pct_currency": round(ltcg_tax, 2),
            "annual_ltcg_exemption_currency": round(annual_ltcg_exemption_currency, 2),
            "ltcg_gain_already_used_currency": round(ltcg_gain_already_used_currency, 2),
            "total_statutory_and_tax_drag_currency": round(total_friction, 2),
            "total_drag_bps": round((total_friction / portfolio_value) * 10000.0, 2),
            "assumptions": [
                "Illustrative cash-equity delivery rates; verify the trade date, exchange, broker, and segment charges.",
                "Assumes Section 111A STCG at 20% and Section 112A LTCG at 12.5% for eligible listed-equity gains after 23 July 2024.",
                "Applies the ₹125,000 Section 112A exemption in aggregate after user-specified prior usage, not separately per asset.",
                "Excludes cess, surcharge, loss set-off/carry-forward, other realized gains, exemptions, and individualized tax treatment.",
                "Holding days use a simplified input-day threshold; exact acquisition/transfer dates govern the real holding period.",
            ],
        }

    @staticmethod
    def compute_sebi_liquidity_adjusted_var(
        base_var_99_currency: float,
        portfolio_df: pd.DataFrame,
        max_participation_rate: float = 0.10,
        *,
        base_spread_cost_pct: float = 0.0015,
        square_root_impact_coefficient: float = 0.012,
    ) -> dict[str, Any]:
        required = {"market_value", "adtv_currency"}
        if not isinstance(portfolio_df, pd.DataFrame) or portfolio_df.empty or not required.issubset(portfolio_df.columns):
            raise ValueError("portfolio_df must include market_value and adtv_currency columns.")
        if not np.isfinite(base_var_99_currency) or base_var_99_currency < 0:
            raise ValueError("base_var_99_currency must be finite and non-negative.")
        if not 0 < max_participation_rate <= 1:
            raise ValueError("max_participation_rate must be in (0, 1].")
        if base_spread_cost_pct < 0 or square_root_impact_coefficient < 0:
            raise ValueError("Impact parameters must be non-negative.")
        values = pd.to_numeric(portfolio_df["market_value"], errors="coerce").to_numpy(dtype=float)
        adtv = pd.to_numeric(portfolio_df["adtv_currency"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(adtv).all() or np.any(values < 0) or np.any(adtv <= 0):
            raise ValueError("Market values must be finite/non-negative and every ADTV must be finite/positive.")
        total_val = float(values.sum())
        if total_val <= 0:
            raise ValueError("Total portfolio market value must be positive.")
        ratio = values / adtv
        impact_pct = base_spread_cost_pct + square_root_impact_coefficient * np.sqrt(ratio)
        slippage = float(np.sum(values * impact_pct))
        days = values / (adtv * max_participation_rate)
        return {
            "base_var_99_currency": round(float(base_var_99_currency), 2),
            "liquidity_slippage_cost_currency": round(slippage, 2),
            "liquidity_adjusted_lvar_99_currency": round(float(base_var_99_currency + slippage), 2),
            "lvar_99_pct_of_portfolio": round(float((base_var_99_currency + slippage) / total_val), 6),
            "max_days_to_liquidate_50pct": round(float(0.5 * days.max()), 2),
            "max_days_to_liquidate_100pct": round(float(days.max()), 2),
            "max_participation_rate": max_participation_rate,
            "method": "Spread plus square-root impact scenario approximation; not a universal SEBI-prescribed L-VaR formula.",
            "assets_above_20pct_of_adtv": int(np.sum(ratio > 0.20)),
        }


class JudgeChaosHeadlineSimulator:
    """Apply a deterministic keyword-severity shock to user-entered hypothetical headlines.

    This does not fetch breaking news, execute FinBERT, estimate new portfolio losses,
    or update an independently calibrated Hawkes process. VaR/ES changes are scenario
    multipliers based on the supplied starting values.
    """
    HIGH_SEVERITY_TOKENS = {
        "war": 0.42, "cyclone": 0.30, "halt": 0.35, "sanction": 0.38, "default": 0.48,
        "crash": 0.45, "rbi": 0.25, "fed": 0.25, "hike": 0.28, "emergency": 0.40,
        "ban": 0.34, "fraud": 0.44, "crisis": 0.40, "inflation": 0.26, "devaluation": 0.38,
        "cyberattack": 0.36, "outage": 0.30, "collapse": 0.46, "downgrade": 0.32, "tariff": 0.30,
    }

    @classmethod
    def execute_flash_crash_trace(
        cls,
        headline: str,
        base_mu_0: float,
        alpha: float,
        beta: float,
        base_var_99_currency: float,
        base_es_99_currency: float,
    ) -> dict[str, Any]:
        if not isinstance(headline, str) or not headline.strip():
            raise ValueError("Please enter a non-empty hypothetical headline.")
        vals = [base_mu_0, alpha, beta, base_var_99_currency, base_es_99_currency]
        if not np.isfinite(vals).all() or base_mu_0 < 0 or alpha < 0 or beta <= 0 or base_var_99_currency < 0 or base_es_99_currency < 0:
            raise ValueError("Rates and VaR/ES inputs must be finite and non-negative; beta must be positive.")
        if alpha >= beta:
            raise ValueError("Baseline Hawkes parameters must satisfy alpha / beta < 1.")
        tokens = re.findall(r"[a-z0-9]+", headline.lower())
        matched = [(word, cls.HIGH_SEVERITY_TOKENS[word]) for word in tokens if word in cls.HIGH_SEVERITY_TOKENS]
        severity_sum = sum(weight for _, weight in matched)
        # Heuristic keyword score, deliberately not represented as calibrated NLP probabilities.
        p_neg = float(np.clip(0.20 + 0.75 * min(severity_sum, 1.0), 0.20, 0.95))
        p_pos = float((1.0 - p_neg) * 0.25)
        p_neu = float((1.0 - p_neg) * 0.75)
        z_shock = -((p_neg - 0.35) / 0.15)
        gamma_news = 0.65
        downside_z = max(0.0, -z_shock)
        mu_shocked = base_mu_0 * (1.0 + gamma_news * downside_z)
        effective_branching = min((alpha * (1.0 + 0.25 * downside_z)) / beta, 0.96)
        tail_multiplier = 1.0 + 0.14 * downside_z / max(1.0 - effective_branching * 0.5, 0.04)
        var_after = base_var_99_currency * tail_multiplier
        es_after = base_es_99_currency * tail_multiplier * 1.06
        return {
            "headline": headline,
            "scenario_type": "User-entered hypothetical; no live headline feed was queried.",
            "sentiment_method": "Uncalibrated keyword-severity heuristic; not FinBERT.",
            "matched_severity_terms": [{"token": w, "severity_weight": weight} for w, weight in matched],
            "heuristic_sentiment_scores": {"P_positive_heuristic": round(p_pos, 3), "P_neutral_heuristic": round(p_neu, 3), "P_negative_heuristic": round(p_neg, 3)},
            "stage2_sentiment_z_shock": round(float(z_shock), 2),
            "stage3_hawkes_telemetry": {
                "baseline_mu_before": round(float(base_mu_0), 4),
                "baseline_mu_after_scenario": round(float(mu_shocked), 4),
                "branching_ratio_after_scenario": round(float(effective_branching), 3),
                "intensity_surge_pct": round(float(((mu_shocked / max(base_mu_0, 1e-8)) - 1.0) * 100.0), 1) if base_mu_0 > 0 else 0.0,
            },
            "stage4_portfolio_impact": {
                "var_99_before": round(float(base_var_99_currency), 2),
                "var_99_after_scenario": round(float(var_after), 2),
                "es_99_before": round(float(base_es_99_currency), 2),
                "es_99_after_scenario": round(float(es_after), 2),
                "additional_tail_capital_at_risk": round(float(es_after - base_es_99_currency), 2),
            },
        }
