"""Expected shortfall wrappers for convenience."""
from risklens.risk.var import historical_var_es

def expected_shortfall(pnl, confidence: float=0.95) -> float:
    return historical_var_es(pnl,confidence).es
