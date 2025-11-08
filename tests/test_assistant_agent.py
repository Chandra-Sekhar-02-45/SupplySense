import pandas as pd
from agents.assistant_agent import AssistantAgent


def make_report():
    return pd.DataFrame([
        {'sku': 'A', 'current_stock': 10, 'forecast_30d': 50},
        {'sku': 'B', 'current_stock': 20, 'forecast_30d': 5},
    ])


def make_sales():
    return pd.DataFrame({
        'date': pd.date_range('2025-01-01', periods=5),
        'sku': ['A', 'A', 'B', 'A', 'B'],
        'units_sold': [5, 6, 1, 7, 2]
    })


def test_promotion_question_returns_promo_advice():
    a = AssistantAgent()
    ans = a.answer_question('Any promotion ideas?', make_report(), make_sales())
    assert 'discount' in ans.lower() or 'bundle' in ans.lower()


def test_reorder_question_returns_reorder_advice():
    a = AssistantAgent()
    ans = a.answer_question('When should I reorder sku A?', make_report(), make_sales())
    assert 'lead' in ans.lower() or 'reorder' in ans.lower()


def test_ambiguous_question_returns_clarifier():
    a = AssistantAgent()
    ans = a.answer_question('I need help', make_report(), make_sales())
    # Clarifier should ask the user to pick promotion/reorder/bundle
    assert 'promotion' in ans.lower() and 'reorder' in ans.lower() and 'bundle' in ans.lower() and 'trend' in ans.lower()


def test_trend_question_returns_metrics():
    a = AssistantAgent()
    ans = a.answer_question('What is the sales trend?', make_report(), make_sales())
    assert 'sales trend summary' in ans.lower() or '30d total units' in ans.lower()
