"""Macro Navigator dashboard. Run locally with:  streamlit run app.py"""
from __future__ import annotations

import json
import os
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from macronav.pipeline import SNAPSHOT, build_snapshot

st.set_page_config(page_title="Macro Navigator", page_icon="🧭", layout="wide")

INK, SLATE, PAPER, LINE = "#15263B", "#5D6B7A", "#F6F7F4", "#D9DED8"
FONT = "Public Sans, -apple-system, system-ui, Helvetica Neue, Arial, sans-serif"
REGIME_COLOR = {1: "#2E7D5B", 2: "#B7791F", 3: "#A63A32", 4: "#2B5BA8"}
VIEW_ORDER = ["Positive", "Leaning positive", "Neutral", "Leaning negative", "Negative", "No view"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&display=swap');
html, body, .stApp, .stMarkdown, button, input, textarea {{ font-family: 'Public Sans', system-ui, sans-serif; }}
.stApp {{ font-variant-numeric: tabular-nums; }}
.block-container {{ padding-top: 3.2rem; max-width: 1180px; }}
.hero {{ border-left: 6px solid var(--rc); padding: 0.4rem 0 0.6rem 1.1rem; margin-bottom: 0.4rem; }}
.hero .asof {{ color: {SLATE}; font-size: 0.92rem; }}
.hero h1 {{ color: var(--rc); font-size: clamp(2.1rem, 6vw, 3.4rem); line-height: 1.05; font-weight: 700;
           letter-spacing: -0.02em; margin: 0.15rem 0 0.35rem 0; padding: 0; }}
.hero .since {{ color: {INK}; font-weight: 600; font-size: 1.02rem; }}
.hero .stance {{ color: {INK}; max-width: 62ch; font-size: 1.05rem; line-height: 1.5; margin-top: 0.5rem; }}
.amber {{ display: inline-block; margin-top: 0.7rem; padding: 0.3rem 0.7rem; border-radius: 6px;
          background: #FBF0D9; color: #7A5212; border: 1px solid #E7C77D; font-weight: 600; font-size: 0.95rem; }}
.demo {{ padding: 0.5rem 0.8rem; border-radius: 6px; background: #EEF2F8; color: #23406E; border: 1px solid #C5D3EA; margin-bottom: 0.8rem; }}
.why {{ color: {SLATE}; font-size: 0.92rem; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data

def _secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


@st.cache_data(ttl=3600, show_spinner=False)
def build_live(key: str | None, demo: bool) -> dict:
    return build_snapshot(key, demo=demo)


def load_saved() -> dict | None:
    if SNAPSHOT.exists():
        try:
            return json.loads(SNAPSHOT.read_text())
        except Exception:
            return None
    return None


fred_key = _secret("FRED_API_KEY")
demo_env = os.environ.get("MACRONAV_DEMO") == "1"

c1, c2, c3 = st.columns([1.1, 1.1, 3])
refresh = c1.button("Refresh now", type="primary", disabled=not fred_key,
                    help="Pulls the latest data and rebuilds every view (about a minute)." if fred_key
                    else "Add FRED_API_KEY to the app's secrets to enable live refresh.")
demo_click = c2.button("Show demo data", help="Synthetic data, for trying the dashboard without keys.")

if refresh:
    with st.spinner("Pulling FRED, market prices and new research..."):
        try:
            st.session_state["snap"] = build_live(fred_key, False)
        except Exception as err:
            st.error(f"Live refresh failed: {err}. Showing the last saved snapshot instead.")
if demo_click or (demo_env and "snap" not in st.session_state and load_saved() is None):
    st.session_state["snap"] = build_live(None, True)

snap = st.session_state.get("snap") or load_saved()
if snap is None:
    st.title("Macro Navigator")
    st.write("No snapshot yet. Run the **Daily snapshot** workflow in GitHub (Actions tab → Run workflow), "
             "press **Refresh now** once your FRED key is in the app secrets, or try **Show demo data**.")
    st.stop()

generated = datetime.fromisoformat(snap["generated_at"]).strftime("%d %b %Y, %H:%M UTC")
c3.caption(f"Data as of {snap['data_as_of']} · built {generated}")
if snap.get("demo"):
    st.markdown('<div class="demo">Demo mode: these numbers are synthetic and describe no real market.</div>', unsafe_allow_html=True)

reg = snap["regime"]
code = reg["code"]
color = REGIME_COLOR[code]
narr = snap["narrative"]
since = pd.Timestamp(reg.get("regime_since")).strftime("%d %b %Y") if reg.get("regime_since") else "n/a"
title = reg["name"].split(" ", 1)[1]
amber_html = ""
if reg.get("amber"):
    n_signs = len([s for s in narr.get("amber_signs", []) if not s.startswith("Amber stays")])
    amber_html = f'<div class="amber">Amber warning: {n_signs} stress sign{"s" if n_signs != 1 else ""} firing, equity risk trimmed</div>'

left, right = st.columns([1.55, 1])
with left:
    st.markdown(f"""<div class="hero" style="--rc:{color}">
      <div class="asof">Economic regime</div>
      <h1>{title}</h1>
      <div class="since">Regime {code}, in place since {since}</div>
      <div class="stance">{narr['stance']}</div>{amber_html}</div>""", unsafe_allow_html=True)
with right:
    total = snap["score"]["total"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=total, number={"suffix": " / 100", "font": {"size": 34, "color": INK}},
        title={"text": f"Macro risk score · {snap['score']['band']}", "font": {"size": 15, "color": SLATE}},
        gauge={"axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": LINE, "tickvals": [0, 25, 45, 65, 80, 100], "tickfont": {"size": 11, "color": SLATE}},
               "bar": {"color": INK, "thickness": 0.28},
               "steps": [{"range": [0, 25], "color": "#DCEBE3"}, {"range": [25, 45], "color": "#E9EFE3"},
                         {"range": [45, 65], "color": "#F6EAD2"}, {"range": [65, 80], "color": "#F1D9D3"},
                         {"range": [80, 100], "color": "#E8C4BD"}],
               "borderwidth": 0}))
    fig.update_layout(height=250, margin=dict(l=34, r=34, t=56, b=12), paper_bgcolor="rgba(0,0,0,0)", font_family=FONT)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

tabs = st.tabs(["Situation", "US sectors and indices", "Global", "History", "Research watch", "Method and data"])


def fmt(v, unit):
    if v is None:
        return "n/a"
    if unit == "%":
        return f"{v:.2f}%"
    if unit == "pp":
        return f"{v:+.2f} pp"
    if unit == "bp":
        return f"{v:.0f} bp"
    if unit == "$":
        return f"${v:,.2f}"
    if unit == "prob":
        return f"{v:.0%}"
    if unit == "share":
        return f"{v:+.0%}"
    if unit == "pts":
        return f"{v:,.0f}"
    return f"{v:.2f}"


# ---------------------------------------------------------------- Situation
with tabs[0]:
    st.markdown(f"#### Where things stand · risk score {snap['score']['total']}/100 ({snap['score']['band'].lower()})")
    for line in narr["situation"]:
        st.markdown(f"- {line}")
    st.markdown("#### Outlook by horizon")
    for k, v in narr["horizons"].items():
        st.markdown(f"**{k}.** {v}")
    st.markdown(f"**Global.** {narr['global']}")
    if narr.get("amber_signs"):
        st.markdown("#### Amber signs")
        for s in narr["amber_signs"]:
            st.markdown(f"- {s}")
    st.markdown("#### What would change the call")
    for t in narr["triggers"]:
        st.markdown(f"- {t}")

    a, b = st.columns([1, 1])
    with a:
        comp = pd.Series(snap["score"]["components"]).sort_values()
        bar = go.Figure(go.Bar(x=comp.values, y=comp.index, orientation="h",
                               marker_color=[("#A63A32" if v == 10 else "#B7791F" if v == 5 else "#2E7D5B") for v in comp.values]))
        bar.update_layout(title="Risk score by component (0, 5 or 10 each)", height=360, margin=dict(l=10, r=10, t=40, b=10),
                          xaxis=dict(range=[0, 10], dtick=5), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_family=FONT)
        st.plotly_chart(bar, width="stretch", config={"displayModeBar": False})
    with b:
        flags = pd.DataFrame([{"Flag": k.replace("_", " ").title(), "On": "Yes" if v else "No"} for k, v in snap["flags"].items()])
        st.markdown("**Model flags now**")
        st.dataframe(flags, hide_index=True, width="stretch", height=360)

    st.markdown("#### Key readings")
    rd = pd.DataFrame([{"Indicator": r["label"], "Latest": fmt(r["value"], r["unit"])} for r in snap["readings"]])
    st.dataframe(rd, hide_index=True, width="stretch")


# ---------------------------------------------------------------- asset tables

def asset_section(groups: list[str], key: str):
    df = pd.DataFrame([a for a in snap["assets"] if a.get("group") in groups and a.get("status") == "ok"])
    if df.empty:
        st.info("No price data for this group in the latest snapshot.")
        return
    if len(groups) > 1:
        pick = st.radio("Show", groups, horizontal=True, key=f"{key}-grp")
        df = df[df.group == pick]
    fig = go.Figure()
    fig.add_shape(type="line", x0=0, x1=0, y0=-100, y1=100, line=dict(color=LINE))
    fig.add_shape(type="line", x0=-100, x1=100, y0=0, y1=0, line=dict(color=LINE))
    fig.add_trace(go.Scatter(x=df.short_score, y=df.medium_score, mode="markers+text", text=df.ticker,
                             textposition="top center", marker=dict(size=11, color=color, opacity=0.85),
                             hovertext=df.name, hoverinfo="text+x+y"))
    for x, y, txt in [(60, 92, "Positive on both"), (-60, 92, "Medium-term fit, weak trend"),
                      (60, -92, "Strong trend, poor fit"), (-60, -92, "Negative on both")]:
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, font=dict(color=SLATE, size=12))
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(title="1-3 month score", range=[-105, 105], zeroline=False),
                      yaxis=dict(title="6-12 month score", range=[-105, 105], zeroline=False), font_family=FONT, showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    view = df[["ticker", "name", "short_view", "medium_view", "long_view", "ret_1m", "ret_3m", "ret_12m",
               "short_score", "medium_score"]].sort_values("medium_score", ascending=False)
    st.dataframe(view, hide_index=True, width="stretch", column_config={
        "ticker": "Ticker", "name": "Name", "short_view": "1-3 months", "medium_view": "6-12 months", "long_view": "3-5+ years",
        "ret_1m": st.column_config.NumberColumn("1m", format="percent"),
        "ret_3m": st.column_config.NumberColumn("3m", format="percent"),
        "ret_12m": st.column_config.NumberColumn("12m", format="percent"),
        "short_score": st.column_config.NumberColumn("Short score", format="%.0f"),
        "medium_score": st.column_config.NumberColumn("Medium score", format="%.0f"),
    })
    pick_t = st.selectbox("Explain a view", df.ticker.tolist(), key=f"{key}-why",
                          format_func=lambda t: f"{t} — {df.set_index('ticker').loc[t, 'name']}")
    r = df.set_index("ticker").loc[pick_t]
    st.markdown(f"<div class='why'><b>1-3 months: {r.short_view}.</b> {r.short_why}<br><b>6-12 months: {r.medium_view}.</b> "
                f"{r.medium_why}<br><b>3-5+ years: {r.long_view}.</b> {r.long_why}</div>", unsafe_allow_html=True)


with tabs[1]:
    asset_section(["US index", "US sector", "US industry"], "us")
with tabs[2]:
    asset_section(["Global"], "gl")
    if snap.get("global_indices"):
        st.markdown("#### Local-currency indices")
        gi = pd.DataFrame(snap["global_indices"])
        st.dataframe(gi, hide_index=True, width="stretch", column_config={
            "ticker": "Ticker", "name": "Index", "level": st.column_config.NumberColumn("Level", format="%.0f"),
            "ret_1m": st.column_config.NumberColumn("1m", format="percent"),
            "ret_12m": st.column_config.NumberColumn("12m", format="percent")})


# ---------------------------------------------------------------- History
with tabs[3]:
    h = pd.DataFrame(snap["history"])
    if h.empty:
        st.info("History builds up as the daily job runs.")
    else:
        h["date"] = pd.to_datetime(h["date"])
        fig = go.Figure()
        runs = (h.regime != h.regime.shift()).cumsum()
        for _, seg in h.groupby(runs):
            fig.add_vrect(x0=seg.date.iloc[0], x1=seg.date.iloc[-1] + pd.Timedelta(days=7),
                          fillcolor=REGIME_COLOR[int(seg.regime.iloc[0])], opacity=0.12, line_width=0)
        fig.add_trace(go.Scatter(x=h.date, y=h.score, name="Risk score", line=dict(color=INK, width=2)))
        fig.add_trace(go.Scatter(x=h.date, y=h.spx, name="S&P 500", yaxis="y2", line=dict(color=SLATE, width=1.2, dash="dot")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis=dict(title="Risk score", range=[0, 100]), yaxis2=dict(title="S&P 500", overlaying="y", side="right", showgrid=False),
                          legend=dict(orientation="h", y=1.08), font_family=FONT)
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Shading: green Goldilocks, ochre Hawkish, brick Breakdown, cobalt Recovery. "
                   "History length is limited by the high-yield spread series (FRED keeps 3 years; the daily job archives more).")
        changes = h[h.regime != h.regime.shift()][["date", "regime", "score"]]
        changes["regime"] = changes.regime.map({1: "R1 Goldilocks", 2: "R2 Hawkish", 3: "R3 Breakdown", 4: "R4 Recovery"})
        st.dataframe(changes.sort_values("date", ascending=False), hide_index=True, width="stretch")


# ---------------------------------------------------------------- Research
with tabs[4]:
    res = snap.get("research") or {}
    items = res.get("items", [])
    if not items:
        st.info("No papers in this snapshot. The research watch runs with live data (not in demo mode).")
    for it in items:
        badge = "**New.** " if it.get("new") else ""
        st.markdown(f"{badge}[{it['title']}]({it['link']})  \n"
                    f"<span class='why'>{it.get('authors') or 'Authors n/a'} · {it['source']} · "
                    f"module: {str(it.get('module', '')).replace('_', ' ')} · matched: {', '.join(it.get('keywords', [])[:5])}</span>",
                    unsafe_allow_html=True)
    if res.get("status"):
        with st.expander("Feed status"):
            st.json(res["status"])


# ---------------------------------------------------------------- Method
with tabs[5]:
    st.markdown("""
**How the call is made.** Eleven flags (inflation, policy, growth, credit, volatility, trend) feed a four-state
machine: Goldilocks, Hawkish, Breakdown, Recovery. Recovery can only follow Breakdown. Inflation states need
month-end persistence; crisis triggers act immediately. The Amber overlay flags rising stress inside the calm states.

**Horizons.** 1-3 months uses trend and momentum adjusted for the regime. 6-12 months uses regime fit, 12-1 month
relative momentum and recession and dollar overlays. 3-5+ years uses the cyclically adjusted earnings yield and is
shown only where a valuation exists.

**Substitutes.** ISM and the Conference Board LEI are licensed; the model uses the Philadelphia Fed index and
building permits instead. Market breadth uses sector ETFs above their 200-day average.

Thresholds live in `config/thresholds.yaml`; the research basis is in `docs/RESEARCH.md`.
This is a research tool, not investment advice.
""")
    probs = snap["data_status"]["problems"]
    if probs:
        st.markdown("**Data problems in this snapshot**")
        st.dataframe(pd.DataFrame([{"Series": k, "Problem": v} for k, v in probs.items()]), hide_index=True, width="stretch")
    else:
        st.markdown(f"All {snap['data_status']['series_checked']} series loaded.")
