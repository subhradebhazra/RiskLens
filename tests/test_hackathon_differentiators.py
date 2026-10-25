import numpy as np
import pandas as pd
import pytest

from risklens.risk.hackathon_differentiators import (
    ContagionNetwork3DEngine,
    ReverseStressTestSolver,
    IndianRegulatoryFrictionEngine,
    JudgeChaosHeadlineSimulator,
)


def fixture_returns(n=260, seed=23):
    rng = np.random.default_rng(seed)
    common = rng.normal(0.0, 0.008, size=(n, 1))
    specific = rng.normal(0.0, 0.009, size=(n, 4))
    vals = 0.6 * common + specific
    return pd.DataFrame(vals, columns=["A.NS", "B.NS", "C.NS", "D.NS"])


def test_tail_dependence_matrix_valid_range_zero_diagonal():
    returns = fixture_returns()
    result = ContagionNetwork3DEngine.compute_tail_dependence_matrix(returns, nu=4.5)
    assert result.shape == (4, 4)
    assert np.isfinite(result).all()
    assert np.all((result >= 0) & (result <= 1))
    assert np.allclose(np.diag(result), 0)
    assert np.allclose(result, result.T)


def test_three_hop_cascade_is_finite_and_plotly_figure_renders():
    returns = fixture_returns()
    cascade = ContagionNetwork3DEngine.simulate_epicenter_cascade(
        returns, np.full(4, 0.25), "B.NS", initial_shock_pct=-0.15, nu=4.5, edge_threshold=0.0
    )
    assert cascade["hop3_final_shock"].shape == (4,)
    assert np.isfinite(cascade["hop3_final_shock"]).all()
    assert np.all(cascade["hop3_final_shock"] <= 0)
    assert np.isclose(cascade["weights"].sum(), 1)
    fig = ContagionNetwork3DEngine.build_3d_plotly_figure(cascade)
    assert len(fig.data) == 2


def test_reverse_stress_solution_meets_loss_target_and_marks_infeasible():
    returns = fixture_returns()
    result = ReverseStressTestSolver.solve_most_probable_breaking_shock(
        returns, np.full(4, 0.25), portfolio_value=10_000_000, target_loss_currency=750_000
    )
    assert result["feasible"] is True
    assert np.isclose(result["actual_loss_pct"], 0.075, atol=1e-5)
    impossible = ReverseStressTestSolver.solve_most_probable_breaking_shock(
        returns, np.full(4, 0.25), portfolio_value=10_000_000, target_loss_currency=8_000_000
    )
    assert impossible["feasible"] is False
    assert impossible["solver_status"] == "infeasible_target"


def test_rebalance_costs_apply_aggregate_ltcg_exemption_once():
    current = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
    target = np.full(5, 0.2)
    gains = np.full(5, 0.14)
    holding_days = np.full(5, 500)
    result = IndianRegulatoryFrictionEngine.compute_rebalancing_tax_and_stt(
        10_000_000, current, target, gains, holding_days
    )
    assert result["total_turnover_currency"] == pytest.approx(3_000_000)
    assert result["stt_currency"] == pytest.approx(3_000)
    assert result["ltcg_realized_gain_estimate_currency"] > 125_000
    assert result["ltcg_taxable_gain_estimate_currency"] == pytest.approx(
        result["ltcg_realized_gain_estimate_currency"] - 125_000, abs=0.02
    )
    assert result["ltcg_tax_12_5pct_currency"] == pytest.approx(
        result["ltcg_taxable_gain_estimate_currency"] * 0.125, abs=0.02
    )


def test_liquidity_adjusted_var_adds_impact_and_validates_adtv():
    portfolio = pd.DataFrame({"market_value": [1_000_000.0, 2_000_000.0], "adtv_currency": [10_000_000.0, 20_000_000.0]})
    result = IndianRegulatoryFrictionEngine.compute_sebi_liquidity_adjusted_var(100_000.0, portfolio)
    assert result["liquidity_adjusted_lvar_99_currency"] > result["base_var_99_currency"]
    assert result["max_days_to_liquidate_100pct"] > 0
    with pytest.raises(ValueError):
        IndianRegulatoryFrictionEngine.compute_sebi_liquidity_adjusted_var(100_000.0, portfolio.assign(adtv_currency=0))


def test_headline_is_explicitly_a_heuristic_and_validates_hawkes_stationarity():
    result = JudgeChaosHeadlineSimulator.execute_flash_crash_trace(
        "Emergency banking cyberattack and default", 0.05, 0.14, 0.32, 342_000, 468_900
    )
    scores = result["heuristic_sentiment_scores"]
    assert result["sentiment_method"] == "Uncalibrated keyword-severity heuristic; not FinBERT."
    assert sum(scores.values()) == pytest.approx(1.0, abs=0.002)
    assert result["stage4_portfolio_impact"]["var_99_after_scenario"] >= 342_000
    with pytest.raises(ValueError):
        JudgeChaosHeadlineSimulator.execute_flash_crash_trace("crash", 0.05, 0.32, 0.14, 342_000, 468_900)
