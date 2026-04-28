"""
app.py  ─  BiasLens AI  ─  Multi-Agent Bias Auditor
Run:  streamlit run app.py
"""

import io
import sys
import os
import numpy as np
import pandas as pd
import streamlit as st 
import json
target_col = st.session_state.get("target_col", None)
sensitive_col = st.session_state.get("sensitive_col", None)
def to_builtin(obj):
    import numpy as np
    
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_builtin(i) for i in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif hasattr(obj, "item"):  # catches numpy scalars
        return obj.item()
    else:
        return obj
# ── path fix so `modules` is always importable ───────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from demo_data       import generate_demo_dataset
from metrics         import compute_all_metrics
from root_cause      import correlation_with_sensitive, class_imbalance_check
from fix_suggestions import generate_fixes
from agents          import run_data_auditor, run_fairness_judge, run_fix_advisor, run_report_narrator, run_chat_assistant

# helper to convert numpy/pandas types to native Python for JSON serialisation
# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BiasLens AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS  ─ dark cyber theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600;700&display=swap');

  /* ── global ── */
  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
      background-color: #0d0f14;
      color: #e2e8f0;
  }
  .stApp { background-color: #0d0f14; }

  /* ── headings ── */
  h1 { font-family: 'Space Mono', monospace; color: #7ee8fa; letter-spacing: -1px; }
  h2 { font-family: 'Space Mono', monospace; color: #a78bfa; font-size: 1.25rem; }
  h3 { color: #94a3b8; font-size: 1rem; font-weight: 600; }

  /* ── metric cards ── */
  .metric-card {
      background: #161b27;
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 20px 24px;
      margin: 6px 0;
  }
  .metric-high  { border-left: 4px solid #f87171; }
  .metric-low   { border-left: 4px solid #34d399; }
  .badge-high   { background:#451a1a; color:#f87171; padding:3px 10px; border-radius:99px; font-size:.75rem; font-weight:700; }
  .badge-low    { background:#052e16; color:#34d399; padding:3px 10px; border-radius:99px; font-size:.75rem; font-weight:700; }
  .badge-medium { background:#1c1917; color:#fb923c; padding:3px 10px; border-radius:99px; font-size:.75rem; font-weight:700; }

  /* ── agent cards ── */
  .agent-card {
      background: linear-gradient(135deg, #161b27 0%, #1a1f2e 100%);
      border: 1px solid #2d3748;
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
  }
  .agent-header {
      display: flex; align-items: center; gap: 10px;
      font-family: 'Space Mono', monospace;
      color: #7ee8fa; font-size: .85rem; font-weight: 700;
      margin-bottom: 12px;
      text-transform: uppercase; letter-spacing: 1px;
  }

  /* ── fix cards ── */
  .fix-card {
      background: #161b27;
      border: 1px solid #2d3748;
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 12px;
  }
  .fix-high   { border-left: 4px solid #f87171; }
  .fix-medium { border-left: 4px solid #fb923c; }
  .fix-low    { border-left: 4px solid #60a5fa; }

  /* ── section separator ── */
  .section-header {
      background: linear-gradient(90deg, #1e2535 0%, transparent 100%);
      border-left: 3px solid #7ee8fa;
      padding: 10px 16px;
      border-radius: 0 8px 8px 0;
      margin: 28px 0 18px 0;
      font-family: 'Space Mono', monospace;
      font-size: .9rem;
      color: #7ee8fa;
      text-transform: uppercase;
      letter-spacing: 2px;
  }

  /* ── sidebar ── */
  [data-testid="stSidebar"] { background-color: #111520; border-right: 1px solid #1e2535; }
  [data-testid="stSidebar"] .stMarkdown { color: #94a3b8; }

  /* ── dataframe ── */
  .stDataFrame { border-radius: 8px; overflow: hidden; }

  /* ── buttons ── */
  .stButton > button {
      background: linear-gradient(135deg, #5b21b6, #7c3aed);
      color: white; border: none; border-radius: 8px;
      font-family: 'Space Mono', monospace; font-size: .8rem;
      padding: 10px 22px; transition: opacity .2s;
  }
  .stButton > button:hover { opacity: .85; }

  /* ── spinner ── */
  .stSpinner > div { border-top-color: #7ee8fa !important; }

  /* ── info / warning boxes ── */
  .stAlert { border-radius: 10px; }
  div[data-testid="stInfo"] { background: #0f172a; border-color: #7ee8fa; }
  /* ── chat quick-question buttons ── */
  div[data-testid="stHorizontalBlock"] .stButton > button {
      background: #1e2535;
      border: 1px solid #3b4a6b;
      color: #94a3b8;
      font-family: 'Inter', sans-serif;
      font-size: .78rem;
      padding: 8px 12px;
      text-align: left;
  }
  div[data-testid="stHorizontalBlock"] .stButton > button:hover {
      background: #263045;
      color: #e2e8f0;
      opacity: 1;
  }
  /* ── chat input ── */
  [data-testid="stChatInput"] textarea {
      background: #161b27;
      border: 1px solid #3b4a6b;
      border-radius: 10px;
      color: #e2e8f0;
  }

</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 BiasLens AI")
    st.markdown("---")
    st.markdown("**Multi-Agent Bias Auditor**")
    st.markdown(
        "Detect, explain, and fix bias in your datasets using fairness "
        "metrics and AI-powered agent analysis."
    )
    st.markdown("---")
    st.markdown("**Agents**")
    st.markdown("🤖 Data Auditor")
    st.markdown("⚖️ Fairness Judge")
    st.markdown("🔧 Fix Advisor")
    st.markdown("📝 Executive Summary ✨")
    st.markdown("---")
    st.markdown("**AI Features**")
    st.markdown("💬 Chat Assistant ✨")
    st.markdown("---")
    st.markdown("**Fairness Threshold**")
    threshold_display = st.metric("Bias Threshold", "0.10", help="DPD / EOD > 0.10 = High Bias")
    st.markdown("---")
    st.caption("BiasLens AI · MVP v1.1 · Groq llama3-8b-8192")


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 32px 0 8px 0'>
  <h1 style='font-size:2.2rem; margin:0'>🔍 BiasLens AI</h1>
  <p style='color:#64748b; font-size:1rem; margin-top:6px'>
    Multi-Agent Bias Auditor · Detect • Explain • Fix
  </p>
</div>
""", unsafe_allow_html=True)

st.divider()


# ─────────────────────────────────────────────────────────────────────────────
# ① UPLOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">① Upload Data</div>', unsafe_allow_html=True)

col_upload, col_demo = st.columns([3, 1])

df = None

with col_upload:
    uploaded = st.file_uploader(
        "Upload a CSV dataset", type=["csv"],
        help="Must contain a binary target column and at least one sensitive attribute.",
    )

with col_demo:
    st.markdown("<br>", unsafe_allow_html=True)
    use_demo = st.button("▶ Load Demo Dataset", use_container_width=True)

if use_demo:
    df = generate_demo_dataset()
    st.success("✅ Demo dataset loaded (300 rows · biased loan data)")
elif uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.success(f"✅ Uploaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")

if df is not None:
    cols = df.columns.tolist()

    target_col = st.selectbox("Select Target Column", cols)
    sensitive_col = st.selectbox("Select Sensitive Column", cols)

    st.session_state["target_col"] = target_col
    st.session_state["sensitive_col"] = sensitive_col

if df is not None:
    with st.expander("📋 Preview — first 5 rows", expanded=True):
        st.dataframe(df.head(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        target_col = st.selectbox(
            "🎯 Target Column (prediction label)",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
        )
    with c2:
        sensitive_options = [c for c in df.columns if c != target_col]
        sensitive_col = st.selectbox(
            "👤 Sensitive Attribute",
            options=sensitive_options,
            index=0,
        )
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_audit = st.button("🚀 Run Full Audit", use_container_width=True)

    # ── guard: target must be binary ──────────────────────────────────────────
    if df[target_col].nunique() > 10:
        st.warning(
            f"⚠️ Column **{target_col}** has {df[target_col].nunique()} unique values. "
            "Fairness metrics work best with a binary target (0/1). Results may be approximate."
        )

else:
    run_audit = False
    st.info("👆 Upload a CSV or click **Load Demo Dataset** to begin.")


# ─────────────────────────────────────────────────────────────────────────────
# Run the full audit pipeline
# ─────────────────────────────────────────────────────────────────────────────
if df is not None and run_audit:

    # ── compute everything ────────────────────────────────────────────────────
    
    with st.spinner("Computing fairness metrics…"):
        metrics     = compute_all_metrics(df, target_col, sensitive_col)
        dpd = metrics.get("demographic_parity", {}) # type: ignore
        eod = metrics.get("equal_opportunity", {}) # type: ignore
        corr_feats  = correlation_with_sensitive(df, sensitive_col, target_col)
        imbalance   = class_imbalance_check(df, sensitive_col)
        fixes       = generate_fixes(metrics, corr_feats, imbalance)
        

    # ── build dataset summary for agents ─────────────────────────────────────
    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(0)
    dataset_summary = {
        "rows":                    int(df.shape[0]),
        "columns":                 int(df.shape[1]),
        "column_names":            df.columns.tolist(),
        "sensitive_col":sensitive_col,
        "target_col":              target_col,
        "sensitive_distribution":  imbalance["distribution"],
        "is_imbalanced":           imbalance["is_imbalanced"],
        "missing_values":          int(df.isnull().sum().sum()),
        "target_positive_rate":    round(float(df[target_col].mean()), 4),
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ② FAIRNESS METRICS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">② Fairness Metrics</div>', unsafe_allow_html=True)

    dpd = metrics["demographic_parity"]
    eod = metrics["equal_opportunity"]

    

    def bias_badge(interp: str) -> str:
        cls = "badge-high" if interp == "High Bias" else "badge-low"
        return f'<span class="{cls}">{interp}</span>'

    def metric_card(title, value, interp, detail):
        card_class = "metric-high" if interp == "High Bias" else "metric-low"
        st.markdown(f"""
        <div class="metric-card {card_class}">
          <div style="font-size:.75rem;color:#64748b;text-transform:uppercase;letter-spacing:1px">{title}</div>
          <div style="font-size:2.2rem;font-family:'Space Mono',monospace;font-weight:700;margin:6px 0">
            {value:.4f}
          </div>
          {bias_badge(interp)}
          <div style="margin-top:10px;font-size:.85rem;color:#94a3b8">{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        rates_str = " | ".join(f"{g}: {r:.1%}" for g, r in dpd["rates"].items())
        metric_card(
            "Demographic Parity Difference",
            dpd["value"],
            dpd["interpretation"],
            f"Positive-outcome rates → {rates_str}",
        )
    with col_m2:
        metric_card(
            "Equal Opportunity Difference",
            eod["value"],
            eod["interpretation"],
            f"Privileged: {eod['privileged']}  ·  Unprivileged: {eod['unprivileged']}",
        )

    # Overall verdict banner
    overall_bias = dpd["interpretation"] == "High Bias" or eod["interpretation"] == "High Bias"
    if overall_bias:
        st.markdown("""
        <div style="background:#2d1515;border:1px solid #f87171;border-radius:10px;
                    padding:14px 20px;margin-top:12px;color:#fca5a5">
          🚨 <strong>Bias Detected</strong> — One or more metrics exceed the 0.10 threshold.
          Scroll down for root-cause analysis and agent recommendations.
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#052e16;border:1px solid #34d399;border-radius:10px;
                    padding:14px 20px;margin-top:12px;color:#6ee7b7">
          ✅ <strong>Low Bias Detected</strong> — Both metrics are within the acceptable threshold.
          Continue monitoring as your dataset evolves.
        </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ③ AGENT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">③ Multi-Agent Analysis</div>', unsafe_allow_html=True)
    st.caption("Three specialised AI agents analyse your dataset independently.")

    agent_tabs = st.tabs(["🤖 Data Auditor", "⚖️ Fairness Judge", "🔧 Fix Advisor", "📝 Executive Summary"])

    with agent_tabs[0]:
        st.markdown('<div class="agent-card"><div class="agent-header">🤖 Agent 1 · Data Auditor</div>', unsafe_allow_html=True)
        with st.spinner("Data Auditor is analysing your dataset…"):
            auditor_output = run_data_auditor(dataset_summary)
        st.markdown(auditor_output)
        st.markdown('</div>', unsafe_allow_html=True)

    with agent_tabs[1]:
        st.markdown('<div class="agent-card"><div class="agent-header">⚖️ Agent 2 · Fairness Judge</div>', unsafe_allow_html=True)
        with st.spinner("Fairness Judge is evaluating metrics…"):
            judge_output = run_fairness_judge(metrics, sensitive_col, target_col)
        st.markdown(judge_output)
        st.markdown('</div>', unsafe_allow_html=True)

    with agent_tabs[2]:
        st.markdown('<div class="agent-card"><div class="agent-header">🔧 Agent 3 · Fix Advisor</div>', unsafe_allow_html=True)
        with st.spinner("Fix Advisor is preparing recommendations…"):
            advisor_output = run_fix_advisor(fixes, metrics)
        st.markdown(advisor_output)
        st.markdown('</div>', unsafe_allow_html=True)

    with agent_tabs[3]:
        st.markdown('<div class="agent-card"><div class="agent-header">📝 Agent 4 · Executive Summary</div>', unsafe_allow_html=True)
        st.caption("Plain-English narrative for non-technical stakeholders — copy-paste into any report.")
        with st.spinner("Narrator is writing the executive summary…"):
            narrator_output = run_report_narrator(dataset_summary, metrics, corr_feats, fixes)
        st.markdown(narrator_output)
        st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ④ ROOT CAUSE ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">④ Root Cause Analysis</div>', unsafe_allow_html=True)

    rc_col1, rc_col2 = st.columns([3, 2])

    with rc_col1:
        st.markdown("**Feature Correlation with Sensitive Attribute**")
        if corr_feats:
            for feat in corr_feats:
                risk   = feat["risk"]
                corr_v = feat["correlation"]
                bar_w  = min(abs(corr_v) * 100, 100)
                bar_c  = {"High": "#f87171", "Medium": "#fb923c", "Low": "#60a5fa"}.get(risk, "#60a5fa")
                badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(risk, "badge-low")
                st.markdown(f"""
                <div class="metric-card" style="margin-bottom:10px">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-weight:600">{feat['feature']}</span>
                    <span class="{badge_cls}">{risk} Risk</span>
                  </div>
                  <div style="margin:8px 0 4px;font-family:'Space Mono',monospace;font-size:1.1rem;color:{bar_c}">
                    r = {corr_v:+.4f}
                  </div>
                  <div style="background:#1e2535;border-radius:4px;height:6px;margin-top:6px">
                    <div style="background:{bar_c};width:{bar_w:.1f}%;height:100%;border-radius:4px"></div>
                  </div>
                  <div style="font-size:.78rem;color:#64748b;margin-top:6px">
                    Potential proxy for <em>{sensitive_col}</em>
                  </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No numeric features found for correlation analysis.")

    with rc_col2:
        st.markdown("**Sensitive Attribute Distribution**")
        dist_data = pd.DataFrame(
            list(imbalance["distribution"].items()),
            columns=[sensitive_col, "proportion"],
        )
        dist_data["proportion"] = dist_data["proportion"] * 100
        st.dataframe(dist_data, use_container_width=True, hide_index=True)

        if imbalance["is_imbalanced"]:
            st.markdown(f"""
            <div style="background:#2d1f0a;border:1px solid #fb923c;border-radius:8px;
                        padding:12px 16px;margin-top:10px;color:#fdba74;font-size:.85rem">
              ⚠️ <strong>Imbalance detected</strong><br>
              '{imbalance['dominant_group']}' dominates at {imbalance['dominant_ratio']*100:.1f}%
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#052e16;border:1px solid #34d399;border-radius:8px;
                        padding:12px 16px;margin-top:10px;color:#6ee7b7;font-size:.85rem">
              ✅ Groups are approximately balanced
            </div>""", unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ⑤ FIX SUGGESTIONS
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⑤ Fix Suggestions</div>', unsafe_allow_html=True)

    priority_order = {"High": 0, "Medium": 1, "Low": 2}
    fixes_sorted   = sorted(fixes, key=lambda x: priority_order.get(x["priority"], 3))

    for fix in fixes_sorted:
        p   = fix["priority"]
        cls = {"High": "fix-high", "Medium": "fix-medium", "Low": "fix-low"}.get(p, "fix-low")
        badge_cls = {"High": "badge-high", "Medium": "badge-medium", "Low": "badge-low"}.get(p, "badge-low")
        st.markdown(f"""
        <div class="fix-card {cls}">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
            <div style="font-weight:700;font-size:1rem">
              {fix['icon']} {fix['title']}
            </div>
            <span class="{badge_cls}" style="white-space:nowrap;margin-left:12px">{p} Priority</span>
          </div>
          <div style="color:#94a3b8;font-size:.875rem;line-height:1.6">{fix['detail']}</div>
        </div>
        """, unsafe_allow_html=True)

    # ─────────────────────────────────────────────────────────────────────────
    # ⑥ DOWNLOAD REPORT
    # ─────────────────────────────────────────────────────────────────────────

    # ─────────────────────────────────────────────────────────────────────────
    # ⑥ AI CHAT ASSISTANT
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">⑥ Ask the AI Assistant</div>', unsafe_allow_html=True)
    st.caption("Ask any question about your audit results. The assistant knows your exact metrics, features, and fixes.")

# Serialise audit context to a compact JSON string — sent as plain text to Groq
import json as _json
if df is not None and run_audit:
    _audit_ctx_str = json.dumps(
        to_builtin({
            "dataset": dataset_summary,
            "metrics": metrics,
            "correlated_features": corr_feats,
            "imbalance": imbalance,
            "fixes": [
                {"title": f["title"], "priority": f["priority"], "detail": f["detail"]}
                for f in fixes
            ],
        }),
        indent=2,
    )

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []    # list of {role, content}

# Suggested questions
st.markdown("**Quick questions:**")
sq_cols = st.columns(3)
suggestions = [
    "Which group is most disadvantaged?",
    "How serious is this bias?",
    "What should I fix first?",
]
for i, suggestion in enumerate(suggestions):
    with sq_cols[i]:
        if st.button(suggestion, key=f"sq_{i}", use_container_width=True):
            st.session_state._pending_question = suggestion

# Display existing chat messages
chat_container = st.container()
with chat_container:
        for msg in st.session_state.chat_history:
            role_label = "🧑 You" if msg["role"] == "user" else "🤖 BiasLens AI"
            bubble_bg  = "#1e2535" if msg["role"] == "user" else "#161b27"
            bubble_br  = "#3b82f6" if msg["role"] == "user" else "#7ee8fa"
            st.markdown(f"""
            <div style="background:{bubble_bg};border-left:3px solid {bubble_br};
                        border-radius:8px;padding:12px 16px;margin:8px 0">
              <div style="font-size:.75rem;color:#64748b;margin-bottom:6px;
                          font-family:'Space Mono',monospace">{role_label}</div>
              <div style="font-size:.9rem;line-height:1.6">{msg["content"]}</div>
            </div>
            """, unsafe_allow_html=True)

# Chat input
user_input = st.chat_input("Ask about your bias audit…")

# Handle suggested question click OR typed input
pending = st.session_state.pop("_pending_question", None)
question = pending or user_input

if question:
    st.session_state.chat_history.append({"role": "user", "content": question})

    with st.spinner("Thinking…"):
        # Build context for chat assistant
        # Ensure variables exist BEFORE chat
        if 'metrics' not in locals():
            metrics = {"status": "Run audit first"}

        if 'corr_feats' not in locals():
            corr_feats = []

        if 'fixes_sorted' not in locals():
            fixes_sorted = []

        # Build context
        _audit_ctx_str = f"""
Metrics:
{metrics}

Correlations:
{corr_feats}

Fixes:
{fixes_sorted}
"""

        # Run assistant
        answer = run_chat_assistant(_audit_ctx_str, question)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
    
    st.rerun()

if st.session_state.chat_history:
    if st.button("🗑 Clear chat", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown('<div class="section-header">⑦ Export Report</div>', unsafe_allow_html=True)
dpd = metrics.get("demographic_parity", {}) if 'metrics' in locals() else {}
eod = metrics.get("equal_opportunity", {}) if 'metrics' in locals() else {}

report_lines = [
    "BiasLens AI — Bias Audit Report",
    "=" * 50,
    f"Dataset : {df.shape[0]} rows × {df.shape[1]} columns" if df is not None else "Dataset: Not loaded",
    f"Target column : {target_col}" if target_col else "Target column: Not selected",
    f"Sensitive attr : {sensitive_col}",
    "",
    "── Fairness Metrics ─────────────────────────────",
   f"Demographic Parity Difference : {dpd.get('value', 0):.4f} [{dpd.get('interpretation', 'N/A')}]",
    f"Equal Opportunity Difference  : {eod.get('value', 0):.4f}  [{eod.get('interpretation', 'N/A')}]",
    "",
    "── Root Cause ───────────────────────────────────",
]
for f in (corr_feats if 'corr_feats' in locals() else []):
    report_lines.append(f"  {f['feature']:20s}  corr={f['correlation']:+.4f}  risk={f['risk']}")


report_lines += [
    "",
    "── Fix Suggestions ──────────────────────────────",
]
for fix in (fixes_sorted if 'fixes_sorted' in locals() else []):
    report_lines.append(f"[{fix['priority']}] {fix['title']}")
    report_lines.append(f"  {fix['detail']}")
    report_lines.append("")

auditor_output = auditor_output if 'auditor_output' in locals() else "Not generated"
judge_output = judge_output if 'judge_output' in locals() else "Not generated"
advisor_output = advisor_output if 'advisor_output' in locals() else "Not generated"

report_lines += [
    "",
    "── Agent Outputs ────────────────────────────────",
  "Data Auditor:",
auditor_output if 'auditor_output' in locals() else "Not generated", 
    "",
    "Fairness Judge:",
   judge_output if 'judge_output' in locals() else "Not generated",
    "",
   "Fix Advisor:",
advisor_output if 'advisor_output' in locals() else "Not generated",
]

report_text = "\n".join(report_lines)

st.download_button(
        label="📥 Download Audit Report (.txt)",
        data=report_text.encode(),
        file_name="biaslens_audit_report.txt",
        mime="text/plain",
        use_container_width=False,
    )
st.caption("Report includes all metrics, root causes, fixes, and agent analyses.")
