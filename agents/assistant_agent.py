from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import requests
import json
from datetime import datetime
from pathlib import Path

from utils.config import GEMINI_API_KEY
OPENAI_API_KEY = None
try:
    from utils.config import OPENAI_API_KEY as _OPENAI
    OPENAI_API_KEY = _OPENAI
except Exception:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class AssistantAgent:
    """Lightweight assistant that generates sales-increase suggestions and supports a simple interactive Q&A loop.

    This is intentionally a rules-based helper that behaves like an LLM for common, actionable prompts
    (promotion ideas, reorder timing, bundling suggestions). If an LLM API key is present in
    `GEMINI_API_KEY`, the assistant will print that a real LLM could be invoked (integration placeholder).
    """

    def __init__(self):
        # Prefer OpenAI if configured, else Gemini (Google) if configured
        # If a Gemini key is present prefer it and disable OpenAI to avoid
        # attempting OpenAI calls when the user intends to use Gemini only.
        self.gemini_key = GEMINI_API_KEY
        self.openai_key = None if self.gemini_key else OPENAI_API_KEY
        # Allow disabling external LLM calls via env (and auto-disable during pytest)
        disable_flag = os.getenv("SUPPLYSENSE_DISABLE_LLM") or os.getenv("PYTEST_CURRENT_TEST")
        self.disable_llm = bool(disable_flag)
        self.have_llm = (not self.disable_llm) and bool(self.gemini_key or self.openai_key)

    def suggest_actions(self, report_df: pd.DataFrame, sales_df: pd.DataFrame) -> List[str]:
        """Return a short list of actionable suggestions derived from the report and recent sales."""
        suggestions: List[str] = []
        if report_df.empty:
            return ["No SKUs found in report to analyze."]

        # Top SKUs by forecast
        top = report_df.sort_values("forecast_30d", ascending=False).head(3)
        for _, row in top.iterrows():
            sku = row["sku"]
            forecast = float(row.get("forecast_30d", 0.0))
            current = float(row.get("current_stock", 0.0))
            if current < forecast * 0.2:
                suggestions.append(
                    f"SKU {sku}: forecast {forecast:.0f} vs current stock {current:.0f} — consider increasing reorder frequency or expediting lead time."
                )
            else:
                suggestions.append(f"SKU {sku}: healthy stock for expected demand; monitor for spikes.")

        # Low performers: low forecast but decent stock -> consider promotions
        low = report_df[report_df["forecast_30d"] < report_df["current_stock"]].sort_values("forecast_30d")
        if not low.empty:
            sample = low.head(3)
            for _, r in sample.iterrows():
                suggestions.append(
                    f"SKU {r['sku']}: current stock ({r['current_stock']:.0f}) exceeds forecast ({r['forecast_30d']:.0f}). Recommend promotion or bundle to free up capital."
                )

        # Quick cross-sell idea using recent sales correlations (very simple proxy)
        try:
            recent = sales_df[sales_df["date"] >= pd.to_datetime(sales_df["date"]).max() - pd.Timedelta(days=30)]
            pairs = (
                recent.groupby(["sku"])['units_sold'].sum().sort_values(ascending=False).head(3).index.tolist()
            )
            if len(pairs) >= 2:
                suggestions.append(
                    f"Cross-sell idea: consider bundling {pairs[0]} with {pairs[1]} based on recent co-demand."
                )
        except Exception:
            # best-effort only
            pass

        if not suggestions:
            suggestions.append("No immediate suggestions generated; data may be too sparse.")

        # Add note about LLM availability
        if self.have_llm:
            suggestions.append(
                "LLM integration is available (GEMINI_API_KEY set). Ask free-text questions for richer, contextual recommendations."
            )
        else:
            suggestions.append(
                "Assistant running in local rule-based mode. Set GEMINI_API_KEY to enable LLM-backed answers."
            )

        return suggestions

    def interactive_loop(self, report_df: pd.DataFrame, sales_df: pd.DataFrame) -> None:
        """Simple REPL: shows suggestions, then accepts user questions until 'exit' or blank.

        Answers are produced from rules; when an LLM key is present this is a placeholder where a
        call to the LLM would be made.
        """
        print("\n--- Assistant suggestions to increase sales ---")
        for s in self.suggest_actions(report_df, sales_df):
            print("-", s)

        print("\nYou can ask follow-up questions (type 'exit' to quit).")
        while True:
            try:
                q = input("Assistant> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting assistant.")
                break
            if not q or q.lower() in {"exit", "quit"}:
                print("Goodbye from assistant.")
                break

            # If an LLM key is present we would route this question to the model.
            if self.have_llm:
                print("[LLM mode] An LLM is configured but external calls are not enabled in this environment.")
                print("(In a deployed setup the assistant would forward your question to the configured LLM API.)")
                continue

            # Very small rule-based answers
            q_low = q.lower()
            if "promotion" in q_low or "discount" in q_low:
                print("Assistant:", "Consider time-limited discounts on low-velocity SKUs and bundle slow movers with top sellers.")
            elif "reorder" in q_low or "stock" in q_low or "lead" in q_low:
                print("Assistant:", "Check SKU lead times and consider breaking larger orders into smaller, more frequent shipments for fast movers.")
            elif "bundle" in q_low or "cross" in q_low:
                print("Assistant:", "Identify SKU pairs with overlapping purchase windows and test a small bundle promotion for 2-4 weeks.")
            else:
                print("Assistant:", "I don't fully understand that request in rule-mode. Try asking about 'promotion', 'reorder', or 'bundle'.")

    def answer_question(self, question: str, report_df: pd.DataFrame, sales_df: pd.DataFrame, api_url_override: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """Return a short answer for a single free-text question in rule-mode or an LLM placeholder.

        This mirrors the interactive behaviour but is safe to call from an API route.
        """
        q_low = (question or "").lower()

        # Early handling for trend/market queries with a lightweight analysis if data present
        if any(k in q_low for k in ("trend", "market", "growth")) and isinstance(sales_df, pd.DataFrame) and not sales_df.empty:
            try:
                return self._build_trend_summary(sales_df, report_df, question)
            except Exception:
                pass  # fall through to normal logic

        if self.have_llm:
            # Prefer OpenAI provider if available
            if self.openai_key:
                try:
                    return self._call_openai(question, report_df, sales_df)
                except Exception as e:
                    fallback = self._rule_answer(q_low)
                    if fallback.startswith("I don't fully understand"):
                        return f"[LLM call failed: {e}] Falling back to local assistant.\n" + self._clarify_question()
                    return f"[LLM call failed: {e}] Falling back to local assistant.\n" + fallback

            if self.gemini_key:
                try:
                    return self._call_gemini(question, report_df, sales_df, api_url_override=api_url_override, model_name=model_name)
                except Exception as e:
                    fallback = self._rule_answer(q_low)
                    if fallback.startswith("I don't fully understand"):
                        return f"[LLM call failed: {e}] Falling back to local assistant.\n" + self._clarify_question()
                    return f"[LLM call failed: {e}] Falling back to local assistant.\n" + fallback

        # If the rule-based answer is ambiguous, return clarifying questions instead
        rule_ans = self._rule_answer(q_low)
        if rule_ans.startswith("I don't fully understand"):
            return self._clarify_question()
        return rule_ans

    # ---------------- Provider implementations ----------------
    def _call_openai(self, question: str, report_df: pd.DataFrame, sales_df: pd.DataFrame) -> str:
        """Call OpenAI Chat Completions (REST) as a fallback provider.

        Uses the OPENAI_API_KEY if present. Sends a concise report summary and the user question.
        """
        if not self.openai_key:
            raise RuntimeError("No OPENAI_API_KEY configured")

        # build prompt / messages
        summary = self._build_summary(report_df)
        messages = [
            {"role": "system", "content": "You are an inventory and sales growth assistant."},
            {"role": "user", "content": summary + "\n\nUser question: " + (question or "Please provide recommendations to increase sales.")},
        ]

        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {self.openai_key}", "Content-Type": "application/json"}
        payload = {"model": "gpt-4o-mini", "messages": messages, "temperature": 0.2, "max_tokens": 512}

        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"OpenAI request failed: {resp.status_code} {resp.text}")
        data = resp.json()
        # parse response
        try:
            return data["choices"][0]["message"]["content"].strip()
        except Exception:
            return str(data)

    def _rule_answer(self, q_low: str) -> str:
        if "promotion" in q_low or "discount" in q_low:
            return "Consider time-limited discounts on low-velocity SKUs and bundle slow movers with top sellers."
        if "reorder" in q_low or "stock" in q_low or "lead" in q_low:
            return "Check SKU lead times and consider breaking larger orders into smaller, more frequent shipments for fast movers."
        if "bundle" in q_low or "cross" in q_low:
            return "Identify SKU pairs with overlapping purchase windows and test a small bundle promotion for 2-4 weeks."
        if "trend" in q_low or "market" in q_low or "sales" in q_low:
            return "Provide sales trend context: ask for 'market trend details' to receive growth %, top movers, and slow SKUs summary."

        return "I don't fully understand that request in rule-mode."

    def _clarify_question(self) -> str:
        """Return a short set of clarifying questions to disambiguate the user's intent.

        Focused on the three common business intents we support in rule-mode: promotion, reorder, bundle.
        """
        clarifying = [
            "I didn't understand your request. Which of these do you mean?",
            "- Promotion: ideas to drive sales (discounts, email campaigns, channels)",
            "- Reorder: procurement timing or reorder-point policy (lead time, MOQ, budget)",
            "- Bundle: combine SKUs to increase AOV or move slow stock",
            "- Trend: analyze sales velocity changes and top/slow movers",
            "Please reply with one of: 'promotion', 'reorder', 'bundle', or 'trend' plus any context (SKUs, budget, time frame).",
        ]
        return "\n".join(clarifying)

    def _build_trend_summary(self, sales_df: pd.DataFrame, report_df: pd.DataFrame, question: str) -> str:
        """Compute simple sales trend metrics.

        Metrics:
        - 30d vs prior 30d total units growth % (if enough history)
        - Top rising SKU (7d avg vs prior 7d avg) and % change
        - Any SKU with declining 7d avg vs prior 7d
        - Current potential stockout count from report_df
        """
        if "date" not in sales_df.columns or "units_sold" not in sales_df.columns:
            return "Insufficient columns to compute trend (need 'date' and 'units_sold')."
        sdf = sales_df.copy()
        sdf['date'] = pd.to_datetime(sdf['date'])
        max_date = sdf['date'].max()
        # Windows
        last30 = sdf[sdf['date'] >= max_date - pd.Timedelta(days=30)]
        prev30 = sdf[(sdf['date'] < max_date - pd.Timedelta(days=30)) & (sdf['date'] >= max_date - pd.Timedelta(days=60))]
        total_last30 = last30['units_sold'].sum()
        total_prev30 = prev30['units_sold'].sum() or 0
        if total_prev30 > 0:
            growth_pct = (total_last30 - total_prev30) / total_prev30 * 100.0
        else:
            growth_pct = float('nan')

        # SKU level 7d vs prior 7d
        last7_cut = max_date - pd.Timedelta(days=7)
        prev7_cut = max_date - pd.Timedelta(days=14)
        last7 = sdf[sdf['date'] >= last7_cut]
        prev7 = sdf[(sdf['date'] < last7_cut) & (sdf['date'] >= prev7_cut)]
        last7_avg = last7.groupby('sku')['units_sold'].mean()
        prev7_avg = prev7.groupby('sku')['units_sold'].mean()
        rising = []
        declining = []
        for sku, val in last7_avg.items():
            prev_val = prev7_avg.get(sku, 0.0)
            if prev_val == 0 and val == 0:
                continue
            if prev_val == 0 and val > 0:
                rising.append((sku, float('inf')))  # newly active
            else:
                change_pct = (val - prev_val) / prev_val * 100.0 if prev_val else float('inf')
                if change_pct >= 5:
                    rising.append((sku, change_pct))
                elif change_pct <= -5:
                    declining.append((sku, change_pct))
        rising_sorted = sorted(rising, key=lambda x: (-x[1] if x[1] != float('inf') else -1e9))
        top_rising = rising_sorted[0] if rising_sorted else None

        # Stockout risk from report
        stockouts = None
        if isinstance(report_df, pd.DataFrame) and not report_df.empty and 'current_stock' in report_df.columns and 'reorder_point' in report_df.columns:
            stockouts = int((report_df['current_stock'] < report_df['reorder_point']).sum())

        parts = ["Sales trend summary:"]
        if not pd.isna(growth_pct):
            parts.append(f"30d total units: {total_last30} (growth vs prior 30d: {growth_pct:.1f}% from {total_prev30}).")
        else:
            parts.append(f"30d total units: {total_last30} (insufficient prior 30d history for growth%).")
        if top_rising:
            pct = top_rising[1]
            if pct == float('inf'):
                parts.append(f"Top rising SKU: {top_rising[0]} (active this week, previously zero).")
            else:
                parts.append(f"Top rising SKU: {top_rising[0]} (+{pct:.1f}% avg units vs prior 7d).")
        if declining:
            decl_list = ", ".join(f"{sku} ({chg:.1f}%)" for sku, chg in declining[:5])
            parts.append(f"Declining SKUs (>=5% drop): {decl_list}.")
        if stockouts is not None:
            parts.append(f"Potential stockouts: {stockouts} SKU(s) below reorder point.")
        parts.append("Query interpreted as a trend analysis. For promotion ideas include the word 'promotion'.")
        return "\n".join(parts)

    def _call_gemini(self, question: str, report_df: pd.DataFrame, sales_df: pd.DataFrame, api_url_override: Optional[str] = None, model_name: Optional[str] = None) -> str:
        """Call a Gemini/Generative API to get a detailed answer.

        This method is intentionally defensive:
        - gated by presence of `GEMINI_API_KEY`
        - uses a reasonable timeout
        - summarizes the dataframes instead of sending raw full tables
        - falls back to rule-based answers on any failure

        It attempts to call a Google Generative API style endpoint by default. You can
        override the endpoint by setting `GEMINI_API_URL` in the environment.
        """
        if self.disable_llm:
            # Explicitly disabled (e.g. during tests); return rule answer
            return self._rule_answer((question or "").lower())

        if not GEMINI_API_KEY:
            raise RuntimeError("No GEMINI_API_KEY configured")

        # Modern Gemini endpoint selection (generateContent). Default model if not provided.
        # Prefer a widely-available default
        model = model_name or os.getenv("GEMINI_MODEL") or "gemini-1.5-flash"
        api_url_override = api_url_override or os.getenv("GEMINI_API_URL")
        candidate_urls: List[str] = []
        if api_url_override:
            candidate_urls.append(api_url_override.rstrip("/"))
        # Provide a small list of model variants to try automatically if first fails.
        model_variants = [model, "gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"]
        for m in model_variants:
            candidate_urls.append(f"https://generativelanguage.googleapis.com/v1/models/{m}:generateContent")
            candidate_urls.append(f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent")

        # Build a concise context summary to include in the prompt
        parts: List[str] = []
        try:
            parts.append(f"Number of SKUs in report: {int(report_df['sku'].nunique())}")
            top_forecast = report_df.sort_values("forecast_30d", ascending=False).head(3)
            parts.append("Top forecasted SKUs:")
            for _, r in top_forecast.iterrows():
                parts.append(f"- {r['sku']}: forecast {int(r['forecast_30d'])}, current {int(r['current_stock'])}")
            low = report_df[report_df['forecast_30d'] < report_df['current_stock']].head(3)
            if not low.empty:
                parts.append("SKUs with excess stock (suggest promotion):")
                for _, r in low.iterrows():
                    parts.append(f"- {r['sku']}: current {int(r['current_stock'])}, forecast {int(r['forecast_30d'])}")
        except Exception:
            # best-effort summarization only
            parts.append("(could not summarize report dataframe)")

        prompt = (
            "You are a helpful inventory and sales growth assistant. "
            "Given the SupplySense report summary and user question, provide 3-5 prioritized, actionable recommendations (each with 1-line rationale)."\
        ) + "\n\n" + "\n".join(parts) + "\n\nUser question: " + (question or "Provide recommendations to increase sales.")

        # New style Gemini content payload
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 512
            }
        }

        # Decide how to authenticate: Google API keys (start with "AIza") are sent as query param.
        params = None
        headers = {"Content-Type": "application/json"}
        if GEMINI_API_KEY.startswith("AIza"):
            params = {"key": GEMINI_API_KEY}
        else:
            # assume OAuth-like bearer token
            headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"

    # Try candidate URLs until one succeeds
        responses = []
        for url in candidate_urls:
            try:
                resp = requests.post(url, json=payload, headers=headers, params=params, timeout=15)
            except Exception as e:
                responses.append({"url": url, "error": str(e)})
                continue

            data_text = resp.text if resp is not None else "(no response)"
            if resp.status_code == 200:
                try:
                    data = resp.json()
                except Exception:
                    data = {"raw": data_text}
                # successful response
                break
            else:
                # Try to extract concise error message
                msg = None
                try:
                    j = resp.json()
                    msg = j.get("error", {}).get("message")
                except Exception:
                    msg = None
                responses.append({"url": url, "status": resp.status_code, "response": (msg or data_text)[:300]})
                data = None
        else:
            # All attempts failed; try to discover available models via ListModels
            available_models: List[str] = []
            for lm_base in ["https://generativelanguage.googleapis.com/v1/models", "https://generativelanguage.googleapis.com/v1beta/models"]:
                try:
                    lm_resp = requests.get(lm_base, headers=headers, params=params, timeout=10)
                    if lm_resp.status_code == 200:
                        data_lm = lm_resp.json()
                        names = [m.get("name", "") for m in data_lm.get("models", []) if isinstance(m, dict)]
                        available_models.extend([n for n in names if n])
                except Exception:
                    pass

            # Try the first discovered model once as a last chance
            if available_models:
                first = available_models[0].split("/")[-1]
                try_url = f"https://generativelanguage.googleapis.com/v1/models/{first}:generateContent"
                try:
                    resp2 = requests.post(try_url, json=payload, headers=headers, params=params, timeout=15)
                    if resp2.status_code == 200:
                        try:
                            data = resp2.json()
                        except Exception:
                            data = {"raw": resp2.text}
                        # parse success below
                    else:
                        responses.append({"url": try_url, "status": resp2.status_code, "response": resp2.text[:300]})
                        data = None
                except Exception as e:
                    responses.append({"url": try_url, "error": str(e)})
                    data = None
            else:
                data = None

            # If still no success, write diagnostics and raise informative error
            try:
                log_dir = Path("logs")
                log_dir.mkdir(exist_ok=True)
                diag = {
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "attempts": responses,
                    "model_sequence": model_variants,
                    "available_models": available_models[:20],
                    "params": {k: v for k, v in (params or {}).items() if k.lower() != "key"},
                    "headers": {k: v for k, v in headers.items() if k.lower() != "authorization"},
                    "payload_summary": {"prompt_length": len(prompt), "max_output_tokens": payload.get("generationConfig", {}).get("maxOutputTokens")},
                }
                (log_dir / "assistant_diag.log").write_text(json.dumps(diag, indent=2))
            except Exception:
                pass

            if data is None:
                raise RuntimeError(
                    "LLM request failed for all tried endpoints. See ./logs/assistant_diag.log for diagnostics (no secret key included)."
                )
        # Google-style generative responses sometimes put text under 'candidates'[0]['content']
        # or under 'output' / 'content'. Try common locations.
        text: Optional[str] = None
        try:
            if isinstance(data, dict) and "candidates" in data and data["candidates"]:
                # Gemini generateContent format
                cand = data["candidates"][0]
                content = cand.get("content") or {}
                parts = content.get("parts") if isinstance(content, dict) else None
                if parts and isinstance(parts, list) and parts:
                    first = parts[0]
                    if isinstance(first, dict):
                        text = first.get("text") or first.get("content")
                    elif isinstance(first, str):
                        text = first
        except Exception:
            text = None

        if not text:
            # last resort: stringify the whole response (safe fallback for debugging)
            text = data.get("candidates", [{}])[0].get("content") if isinstance(data, dict) else str(data)

        return text or "(empty response from LLM)"
