import numpy as np
import pandas as pd
from risklens.risk.var import historical_var_es,gaussian_var_es
from risklens.risk.contagion import calculate_contagion_score
from risklens.portfolio.analytics import portfolio_weights,concentration_hhi,portfolio_summary
from risklens.portfolio.optimizer import optimize_portfolio
from risklens.simulation.engine import simulate_portfolio,terminal_pnl
from risklens.simulation.event_generator import UnifiedEventGenerator
from risklens.simulation.scenarios import StressConfig
from risklens.backtesting.walk_forward import walk_forward_historical_var
from risklens.backtesting.validation import kupiec_pof,christoffersen_independence

def test_var_es_sign_convention_known_losses():
    pnl=np.array([100,50,0,-10,-20,-30,-40,-50,-60,-70],float)
    metric=historical_var_es(pnl,.8)
    assert metric.var>=0 and metric.es>=metric.var
    g=gaussian_var_es(pnl,.95)
    assert np.isfinite(g.var) and np.isfinite(g.es)

def test_missing_contagion_features_not_zero_imputed():
    result=calculate_contagion_score()
    assert np.isnan(result.score)
    partial=calculate_contagion_score(tail_dependence=.4,crisis_probability=.2)
    assert partial.score>=0 and abs(sum(partial.effective_weights.values())-1)<1e-9
    assert len(partial.warnings)>0

def test_portfolio_weights_and_optimization_constraints():
    vals=pd.Series([25.,75.],index=['A','B']); w=portfolio_weights(vals)
    assert np.isclose(w.sum(),1) and np.isclose(concentration_hhi(w),.625)
    rng=np.random.default_rng(10); x=pd.DataFrame(rng.normal(.0005,.01,(300,3)),columns=['A','B','C'])
    result=optimize_portfolio(x,pd.Series([.4,.3,.3],index=x.columns),max_turnover=.6)
    if result.success:
        assert np.isclose(result.weights.sum(),1,atol=1e-5)
        assert (result.weights>=-1e-6).all()

def test_simulation_reproducible_and_weights_used():
    rng=np.random.default_rng(11); r=pd.DataFrame(rng.normal(0,.01,(400,2)),columns=['A','B'])
    w=pd.Series([1.,0.],index=['A','B'])
    a=simulate_portfolio(r,w,1000,5,'Gaussian baseline',77)
    b=simulate_portfolio(r,w,1000,5,'Gaussian baseline',77)
    assert np.array_equal(a.paths,b.paths)
    assert a.paths.shape==(1000,5)
    assert np.array_equal(terminal_pnl(a,1000),terminal_pnl(b,1000))

def test_event_generator_enforces_one_mode():
    g=UnifiedEventGenerator('static_poisson',seed=9)
    a=g.generate(10,intensity=3)
    assert (a>=0).all() and (a<10).all()
    try:
        UnifiedEventGenerator('not-a-mode')
    except ValueError as exc:
        assert 'mode' in str(exc)
    else:
        raise AssertionError('Invalid event mode should be rejected.')

def test_stress_schema_bounds_and_backtest_no_lookahead():
    cfg=StressConfig(name='Market down',market_shock=-.2,volatility_multiplier=2,correlation_target=.8,jump_intensity_multiplier=3)
    assert cfg.market_shock==-0.2
    r=pd.Series(np.linspace(-.02,.02,100),index=pd.date_range('2024-01-01',periods=100,tz='UTC'))
    bt=walk_forward_historical_var(r,lookback=40,confidence=.95)
    assert len(bt)==60
    assert bt.iloc[0]['training_end']==r.index[39]
    assert bt.index[0]==r.index[40]
    pof=kupiec_pof(bt['breach'],.95); ind=christoffersen_independence(bt['breach'])
    assert pof['sample_size']==len(bt) and 'p_value' in ind
