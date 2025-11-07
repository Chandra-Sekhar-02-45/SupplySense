import pandas as pd
from utils.data_loader import load_sales_csv


def test_load_sales_csv(tmp_path):
	p = tmp_path / "sales.csv"
	pd.DataFrame({
		'date': pd.date_range('2025-01-01', periods=3, freq='D'),
		'sku': ['A','A','A'],
		'units_sold': [1,2,3]
	}).to_csv(p, index=False)
	df = load_sales_csv(str(p))
	assert set(['date','sku','units_sold']).issubset(df.columns)
