"""
agents.py
Multi-agent bias auditor — powered by Groq API (llama-3.1-8b-instant).

Set GROQ_API_KEY in your environment before running.
Falls back to rule-based output if the API is unavailable.
"""

import json
import os

import requests

def to_builtin(obj):
    import numpy as np

    # dict → fix keys + values
    if isinstance(obj, dict):
        return {str(to_builtin(k)): to_builtin(v) for k, v in obj.items()}

    # list / tuple
    if isinstance(obj, (list, tuple)):
        return [to_builtin(x) for x in obj]

    # numpy scalars → python
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()

    # numpy arrays → list
    if hasattr(obj, "tolist"):
        return obj.tolist()

    return obj
# ── Config ────────────────────────────────────────────────────────────────────
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.1-8b-instant-2048"  # Use a smaller model for faster responses in this demo


# ── Utility ───────────────────────────────────────────────────────────────────

def clean_for_json(obj):
    """
    Recursively converts an object so it is safe to pass to json.dumps.
    Handles numpy scalars, pandas types, sets, and other non-serialisable types.
    """
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_for_json(i) for i in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, str):
        return obj
    # Handle numpy / pandas scalar types without importing numpy
    type_name = type(obj).__name__
    if type_name in ("int64", "int32", "int16", "int8",
                     "uint64", "uint32", "uint16", "uint8"):
        return int(obj)
    if type_name in ("float64", "float32", "float16"):
        return float(obj)
    if type_name in ("bool_",):
        return bool(obj)
    if type_name in ("ndarray",):
        return [clean_for_json(i) for i in obj.tolist()]
    return str(obj)


# ── Core LLM call ─────────────────────────────────────────────────────────────

def call_llm(system_prompt: str, user_prompt: str) -> str:
    """
    Sends a request to the Groq chat completions endpoint.
    Returns the generated text string.
    On any error returns an error string — never raises.
    """
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        return "ERROR: GROQ_API_KEY environment variable is not set."

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature":  0.4,
        "max_tokens":   600,
        "top_p":        0.9,
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except requests.exceptions.HTTPError as e:
        return f"ERROR: Groq API HTTP error — {e}"
    except requests.exceptions.ConnectionError:
        return "ERROR: Could not connect to Groq API. Check your internet connection."
    except requests.exceptions.Timeout:
        return "ERROR: Groq API request timed out."
    except (KeyError, IndexError):
        return "ERROR: Unexpected response format from Groq API."
    except Exception as e:
        return f"ERROR: {e}"


# ── Fallback helpers (rule-based, no API needed) ──────────────────────────────

def _fallback_auditor(summary: dict) -> str:
    rows     = summary.get("rows", "?")
    cols     = summary.get("columns", "?")
    sens     = summary.get("sensitive_col", "?")
    dist     = summary.get("sensitive_distribution", {})
    dist_str = ", ".join(f"{k}: {v*100:.1f}%" for k, v in dist.items())
    return (
        f"**Dataset Overview**\n"
        f"• {rows} records | {cols} features\n"
        f"• Sensitive attribute '{sens}' distribution: {dist_str}\n"
        f"• Missing values: {summary.get('missing_values', 0)}\n\n"
        f"**Finding:** The dataset shows a class imbalance in the sensitive "
        f"attribute which may propagate downstream bias."
    )


def _fallback_judge(metrics: dict) -> str:
    dpd     = metrics.get("demographic_parity", {})
    eod     = metrics.get("equal_opportunity",  {})
    dpd_val = dpd.get("value", 0)
    eod_val = eod.get("value", 0)
    priv    = dpd.get("privileged",   "Group A")
    unpriv  = dpd.get("unprivileged", "Group B")
    verdict = "FAIL" if abs(dpd_val) > 0.1 or abs(eod_val) > 0.1 else "PASS"
    return (
        f"**Fairness Verdict: {verdict}**\n\n"
        f"• Demographic Parity Difference: {dpd_val:.4f} — "
        f"{dpd.get('interpretation', '?')}\n"
        f"• Equal Opportunity Difference:  {eod_val:.4f} — "
        f"{eod.get('interpretation', '?')}\n\n"
        f"**Analysis:** '{priv}' receives favourable outcomes at a significantly "
        f"higher rate than '{unpriv}'. Both metrics exceed the 0.10 fairness "
        f"threshold, confirming systemic bias in outcome distribution."
    )


def _fallback_advisor(fixes: list) -> str:
    lines = ["**Recommended Remediation Plan**\n"]
    for i, f in enumerate(fixes[:3], 1):
        icon   = f.get("icon", "•")
        prior  = f.get("priority", "")
        title  = f.get("title", "")
        detail = f.get("detail", "")[:120]
        lines.append(f"{i}. {icon} [{prior}] **{title}**")
        lines.append(f"   {detail}...")
    lines.append(
        "\n**Implementation Order:** Start with data-level fixes (resampling) "
        "before algorithmic constraints for best results."
    )
    return "\n".join(lines)


def _fallback_narrator(summary: dict, metrics: dict, fixes: list) -> str:
    dpd    = metrics.get("demographic_parity", {})
    dpd_v  = dpd.get("value", 0)
    priv   = dpd.get("privileged",   "one group")
    unpriv = dpd.get("unprivileged", "another group")
    sev    = "significant" if abs(dpd_v) > 0.15 else "moderate"
    top_fix = fixes[0]["title"] if fixes else "dataset rebalancing"
    return (
        f"A bias audit was conducted on a dataset of "
        f"{summary.get('rows', '?')} records examining outcomes for "
        f"'{summary.get('sensitive_col', '?')}'. The analysis revealed "
        f"{sev} disparity in how the system treats different groups.\n\n"
        f"The group identified as '{priv}' receives favourable outcomes at a "
        f"measurably higher rate than '{unpriv}', with a Demographic Parity "
        f"Difference of {dpd_v:.3f}. This gap exceeds the accepted fairness "
        f"threshold of 0.10, indicating the system is not treating all groups "
        f"equally.\n\n"
        f"The audit recommends immediate action starting with {top_fix.lower()}. "
        f"Without intervention, continued use of this system risks perpetuating "
        f"and amplifying existing inequalities — action should be taken before "
        f"deployment."
    )


# ── Public agent functions ────────────────────────────────────────────────────

from groq import Groq
import os

client = Groq(api_key=("GROQ_API_KEY"))

def call_llm(system, user):
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"ERROR: {str(e)}"
    
def run_chat_assistant(context, question):
    try:
    
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are an AI bias analysis assistant."},
                {"role": "user", "content": context + "\n\nQuestion: " + question}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"AI explanation unavailable, showing rule-based output. (ERROR: {str(e)})"
    """
    Interactive Q&A assistant.
    context : serialised audit results (JSON string or plain text)
    question: user's natural-language question
    """
    system = (
        "You are a bias auditing assistant. "
        "The user has run a fairness audit. Answer their question using "
        "only the audit context provided. Be concise (max 150 words), "
        "use markdown formatting, and never invent numbers."
    )
    user = f"Audit context:\n{context}\n\nQuestion: {question}"
    result = call_llm(system, user)
    if result.startswith("ERROR:"):
        return (
            "AI explanation unavailable, showing rule-based output. "
            f"({result})"
        )
    return result


def run_data_auditor(summary: dict) -> str:
    """
    Agent 1 — Data Auditor.
    Accepts the dataset summary dict (as used by app.py).
    """
    system = (
        "You are a data quality expert specialising in bias auditing. "
        "Analyse the dataset summary and give a concise audit report. "
        "Use bullet points. Be direct and technical. Max 200 words."
    )
    user = (
        f"Dataset summary:\n{json.dumps(clean_for_json(summary), indent=2)}\n\n"
        "Provide: (1) a brief overview, (2) any data quality red flags, "
        "(3) initial bias risk signals you observe."
    )
    result = call_llm(system, user)
    if result.startswith("ERROR:"):
        return _fallback_auditor(summary)
    return result


def run_fairness_judge(metrics: dict, sensitive_col: str = "", target_col: str = "") -> str:
    """
    Agent 2 — Fairness Judge.
    Accepts metrics dict + optional column names (as used by app.py).
    """
    system = (
        "You are a fairness auditor trained in algorithmic ethics. "
        "Review the fairness metrics and issue a clear verdict. "
        "Use the 0.10 threshold: above = High Bias, below = Low Bias. "
        "Be concise and structured. Max 200 words."
    )
    safe_metrics = clean_for_json(metrics)
    
    user = (
        f"Sensitive attribute: {sensitive_col}\n"
        f"Target column: {target_col}\n"
        f"Fairness metrics:\n{(json.dumps(to_builtin(metrics), indent=2))}\n\n"
        "Provide: (1) overall fairness verdict (PASS/FAIL), "
        "(2) which groups are disadvantaged, (3) severity assessment."
    )
    result = call_llm(system, user)
    if result.startswith("ERROR:"):
        return _fallback_judge(metrics)
    return result

def run_fix_advisor(fixes: list, metrics: dict = None) -> str:
    """
    Agent 3 — Fix Advisor.
    Accepts fixes list + metrics dict (as used by app.py).
    """
    if metrics is None:
        metrics = {}
    system = (
        "You are a machine-learning fairness engineer. "
        "Given bias fix suggestions, create a clear, prioritised action plan "
        "for the development team. Be practical. Max 200 words."
    )
    fixes_summary = [
        {"title": f.get("title", ""), "priority": f.get("priority", "")}
        for f in fixes
    ]
    dpd = abs(metrics.get("demographic_parity", {}).get("value", 0))
    eod = abs(metrics.get("equal_opportunity",  {}).get("value", 0))
    user = (
        f"DPD={dpd:.4f}, EOD={eod:.4f}\n"
        f"Available fixes:\n{json.dumps(fixes_summary, indent=2)}\n\n"
        "Provide: (1) top 3 highest-impact actions, "
        "(2) implementation order, (3) expected improvement."
    )
    result = call_llm(system, user)
    if result.startswith("ERROR:"):
        return _fallback_advisor(fixes)
    return result


def run_report_narrator(
    summary: dict,
    metrics: dict,
    corr_feats: list,
    fixes: list,
) -> str:
    """
    Agent 4 — Report Narrator.
    Plain-English executive summary for non-technical stakeholders.
    Accepts (summary, metrics, corr_feats, fixes) as used by app.py.
    """
    system = (
        "You are a senior AI ethics consultant writing for a non-technical "
        "executive audience. Summarise a bias audit clearly and compellingly. "
        "Avoid jargon. Use plain language. No bullet points — write flowing "
        "prose. Max 220 words. End with one clear sentence about urgency."
    )
    dpd     = metrics.get("demographic_parity", {})
    eod     = metrics.get("equal_opportunity",  {})
    top_fix = fixes[0]["title"] if fixes else "dataset rebalancing"
    feat_names = [f["feature"] for f in corr_feats if "feature" in f]
    user = (
        f"Dataset: {summary.get('rows', '?')} records, "
        f"sensitive attribute: '{summary.get('sensitive_col', '?')}', "
        f"target: '{summary.get('target_col', '?')}'.\n"
        f"Demographic Parity Difference: {dpd.get('value', 0):.4f} "
        f"({dpd.get('interpretation', '?')}).\n"
        f"Equal Opportunity Difference: {eod.get('value', 0):.4f} "
        f"({eod.get('interpretation', '?')}).\n"
        f"Privileged group: {dpd.get('privileged', '?')}, "
        f"Unprivileged: {dpd.get('unprivileged', '?')}.\n"
        f"Top correlated features: {feat_names}.\n"
        f"Top recommended fix: {top_fix}.\n\n"
        "Write a 3-paragraph executive summary: "
        "(1) what was found, (2) who is affected and how, "
        "(3) what should be done."
    )
    result = call_llm(system, user)
    if result.startswith("ERROR:"):
        return _fallback_narrator(summary, metrics, fixes)
    return result
