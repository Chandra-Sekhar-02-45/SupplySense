import pandas as pd
from agents.safety_stock_agent import SafetyStockAgent


def test_safety_outputs():
	df = pd.DataFrame({
		'date': pd.date_range('2025-01-01', periods=10, freq='D'),
		'sku': ['A'] * 10,
		'units_sold': [10]*10
	})
	agent = SafetyStockAgent(default_lead_time_days=5, service_level=0.95)
	out = agent.compute(df)
	assert {'sku','lead_time_days','safety_stock','reorder_point'}.issubset(out.columns)
	assert len(out) == 1
