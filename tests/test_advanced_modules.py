import numpy as np
import pandas as pd
import pytest
from risklens.models.hamilton_regime import HamiltonRegimeSwitchingModel
from risklens.simulation.unified_event_generator import UnifiedEventGenerator
from risklens.simulation.unified_engine import UnifiedMonteCarloEngine
from risklens.portfolio.institutional_optimizer import InstitutionalPortfolioOptimizer
from risklens.risk.crisis_replay import StressTestingEngine
from risklens.backtesting.walk_forward_suite import WalkForwardBacktester
from risklens.alerts.engine import ConfigurableAlertEngine, AlertRule

def sample_returns(n=320,m=3,seed=7):
    rng=np.random.default_rng(seed); return pd.DataFrame(rng.normal(.0003,.012,(n,m)),columns=[f'A{i}' for i in range(m)])

def test_hamilton_wrapper_returns_probabilities_or_explicit_fallback():
    r=pd.Series(np.random.default_rng(1).normal(0,.01,160))
    model=HamiltonRegimeSwitchingModel(); d=model.fit(r)
    assert len(model.filtered_probs)==160
    assert np.allclose(model.filtered_probs.sum(axis=1),1,atol=1e-5)
    assert d['method_used']

def test_ogata_events_and_jump_counts_reproducible():
    a=UnifiedEventGenerator(seed=2); b=UnifiedEventGenerator(seed=2)
    x=a.simulate_jump_returns(10,5,'hawkes_self_exciting'); y=b.simulate_jump_returns(10,5,'hawkes_self_exciting')
    assert np.array_equal(x['jump_counts'],y['jump_counts'])
    assert np.isfinite(x['jump_returns']).all()

def test_monte_carlo_seeded_and_finite():
    r=sample_returns(); e=UnifiedEventGenerator(seed=5)
    # class supports at least 1,000 paths by design
    a=UnifiedMonteCarloEngine(1000,5,11).run_portfolio_simulation(r,[.4,.35,.25],100000,'merton_only',event_gen=e)
    assert np.isfinite(a['var_99_currency']) and a['n_paths']==1000

def test_cvar_optimizer_returns_valid_weights():
    r=sample_returns(); opt=InstitutionalPortfolioOptimizer(r)
    out=opt.optimize([.4,.35,.25],method='cvar_es',max_weight=.8,max_turnover=.8)
    w=np.asarray(out['optimized_weights']); assert np.isfinite(w).all() and np.isclose(w.sum(),1)

def test_crisis_proxy_is_explicitly_labeled():
    p=pd.DataFrame({'ticker':['A','B'],'market_value':[100.,200.],'sector':['Technology','Financials']})
    out=StressTestingEngine.run_historical_crisis_replay(p,None,'2020_COVID_Liquidity_Crash')
    assert 'proxy' in out['replay_mode'].lower() and out['total_initial_value']==300

def test_walk_forward_backtest_and_alert_thresholds():
    r=pd.Series(np.random.default_rng(3).normal(0,.01,380))
    result=WalkForwardBacktester(.95,300).run_ablation_suite(r)
    assert len(result)>=4 and 'Kupiec_p_value' in result
    alerts=ConfigurableAlertEngine([AlertRule('x','metric','Metric',1.,'>')]).evaluate_telemetry({'metric':2.})
    assert len(alerts)==1 and alerts[0]['observed_value']==2.
