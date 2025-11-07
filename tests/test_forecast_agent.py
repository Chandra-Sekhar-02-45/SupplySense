import pandas as pd
from agents.forecast_agent import ForecastAgent


def test_forecast_shape():
	df = pd.DataFrame({
		'date': pd.date_range('2025-01-01', periods=10, freq='D'),
		'sku': ['A'] * 10,
		'units_sold': [5,6,7,8,9,10,11,12,13,14]
	})
	agent = ForecastAgent(window=3)
	out = agent.forecast(df, horizon_days=5)
	assert len(out) == 5
	assert set(out.columns) == {'sku','date','forecast_units','model_used'}
	assert (out['forecast_units'] >= 0).all()
