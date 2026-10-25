import numpy as np
import pandas as pd
from risklens.features.returns import calculate_returns,annualized_volatility,drawdown_series
from risklens.features.volatility import parkinson_variance,garman_klass_variance
from risklens.data.validation import normalize_ticker,validate_price_data,detect_stale_data
from risklens.models.merton import flag_jump_candidates,fit_merton
from risklens.models.heston import fit_heston,simulate_heston
from risklens.models.bates import fit_bates
from risklens.models.regime_switching import fit_regimes
from risklens.models.student_t_copula import fit_student_t_copula
from risklens.models.hawkes import fit_hawkes,simulate_ogata
from risklens.models.pca_absorption import absorption_ratio


def demo_returns(n=500,seed=7):
    rng=np.random.default_rng(seed); return pd.Series(rng.normal(0.0003,0.01,n),index=pd.date_range('2024-01-01',periods=n,freq='B',tz='UTC'))

def test_returns_and_drawdown():
    prices=pd.Series([100,110,99,108.9])
    r=calculate_returns(prices)
    assert np.allclose(r.to_numpy(),[.1,-.1,.1])
    dd=drawdown_series(r,input_is_returns=True)
    assert np.isclose(dd.min(),-.1)
    assert np.isfinite(annualized_volatility(r))

def test_ticker_normalization_and_price_validation():
    assert normalize_ticker('reliance','NSE')=='RELIANCE.NS'
    assert normalize_ticker('AAPL')=='AAPL'
    idx=pd.to_datetime(['2024-01-02','2024-01-01','2024-01-01'],utc=True)
    df=pd.DataFrame({'Open':[2,1,1],'High':[3,2,2],'Low':[1,1,1],'Close':[2,1,1]},index=idx)
    out,w=validate_price_data(df)
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert any('duplicate' in x.lower() for x in w)
    assert not detect_stale_data(out,max_age_days=365*10)

def test_range_estimators():
    f=pd.DataFrame({'Open':[10.,11.],'High':[12.,12.],'Low':[9.,10.],'Close':[11.,11.5]})
    assert (parkinson_variance(f)>=0).all()
    assert (garman_klass_variance(f)>=0).all()

def test_merton_candidate_flags_and_fit_is_honest():
    r=demo_returns(); r.iloc[100]=-0.15
    flags=flag_jump_candidates(r,threshold=3)
    assert flags.loc[r.index[100]]
    fit=fit_merton(r,threshold=3)
    assert fit.sample_size==len(r)
    assert isinstance(fit.converged,bool)
    assert 'physical-measure' in fit.convention
    short=fit_merton(r.iloc[:10])
    assert not short.converged

def test_heston_nonnegative_variance_and_bates_fallback():
    r=demo_returns()
    fit=fit_heston(r)
    assert fit.converged and fit.theta>0 and fit.xi>=0
    sim=simulate_heston(fit,100,10,seed=11)
    assert sim.shape==(100,10) and np.isfinite(sim).all()
    assert (np.all(sim<10))
    b=fit_bates(r.iloc[:80])
    assert b.selected_model in {'component diagnostics only (not jointly calibrated)','Merton-only fallback','Heston-only fallback','Gaussian baseline fallback'}
    assert any('joint Bates' in w for w in b.warnings)

def test_regime_probabilities_normalize():
    fit=fit_regimes(demo_returns())
    assert fit.probabilities.shape[1]==4
    assert np.allclose(fit.probabilities.sum(axis=1),1,atol=1e-6)
    assert fit.transition_matrix.shape==(4,4)
    assert np.allclose(fit.transition_matrix.sum(axis=1),1)

def test_student_copula_positive_definite_and_tail_dependence():
    rng=np.random.default_rng(5); a=rng.normal(size=400); b=.65*a+rng.normal(size=400)*.7
    frame=pd.DataFrame({'A':a,'B':b})
    fit=fit_student_t_copula(frame)
    assert fit.converged
    assert np.linalg.eigvalsh(fit.correlation).min()>0
    assert fit.lower_tail_dependence.shape==(2,2)
    assert fit.lower_tail_dependence[0,1]>0

def test_hawkes_stationarity_and_ogata():
    rng=np.random.default_rng(8); t=np.sort(rng.uniform(0,200,60))
    fit=fit_hawkes(t,200)
    if fit.converged:
        assert fit.branching_ratio<1
        sim=simulate_ogata(fit,5,seed=1)
        assert (sim>=0).all() and (sim<5).all()

def test_pca_absorption_in_range():
    rng=np.random.default_rng(9); frame=pd.DataFrame(rng.normal(size=(180,5)),columns=list('ABCDE'))
    fit=absorption_ratio(frame,top_k=2,window=60)
    assert 0<=fit.ratio<=1
    assert len(fit.rolling)>0
