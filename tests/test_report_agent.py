import pandas as pd
from agents.report_agent import ReportAgent


def test_report_columns():
	current = pd.DataFrame({'sku':['A'], 'current_stock':[5]})
	forecast = pd.DataFrame({
		'sku':['A']*3,
		'date': pd.date_range('2025-02-01', periods=3, freq='D'),
		'forecast_units':[2,2,2],
		'model_used':['SMA7']*3
	})
	safety = pd.DataFrame({'sku':['A'], 'reorder_point':[12], 'safety_stock':[3]})
	agent = ReportAgent(pack_size_default=1)
	out = agent.build(current, forecast, safety, horizon_days=3)
	assert set(['sku','current_stock','reorder_point','safety_stock','forecast_30d','recommended_order_qty']).issubset(out.columns)
