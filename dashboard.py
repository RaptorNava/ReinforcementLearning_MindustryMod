"""
dashboard.py — Streamlit dashboard for Mindustry RL Agent
Run: streamlit run dashboard.py

Dependencies:
    pip install streamlit plotly pandas
"""

import base64
import json
import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ═══════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════
LOG_CSV   = "training_log.csv"
CKPT_PATH = "mindustry_agent.pt"
MEDIA_DIR = "media"

# Colors — ONLY 6-character hex or rgba() for Plotly!
CLR_BG      = "#0d0f14"
CLR_CARD    = "#13161e"
CLR_BORDER  = "#1e2330"
CLR_ACCENT  = "#00e5ff"
CLR_ACCENT2 = "#ff6b35"
CLR_ACCENT3 = "#7fff6b"
CLR_TEXT    = "#c8d0e0"
CLR_MUTED   = "#4a5568"

# rgba versions for Plotly (no alpha in hex!)
PLOT_ACCENT      = "rgba(0,229,255,1)"
PLOT_ACCENT_DIM  = "rgba(0,229,255,0.25)"
PLOT_ACCENT2     = "rgba(255,107,53,1)"
PLOT_ACCENT3     = "rgba(127,255,107,1)"
PLOT_ACCENT3_DIM = "rgba(127,255,107,0.25)"
PLOT_BG          = "rgba(0,0,0,0)"
PLOT_GRID        = "rgba(30,35,48,1)"
PLOT_SURFACE     = "#0a0c11"


# ═══════════════════════════════════════════════════
#  STYLES & HELPERS
# ═══════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Orbitron:wght@400;700;900&display=swap');

      .stApp, .stApp > div {{
          background-color: {CLR_BG} !important;
          color: {CLR_TEXT} !important;
          font-family: 'Rajdhani', sans-serif;
      }}
      section[data-testid="stSidebar"] {{ background-color: {CLR_BG} !important; }}
      h1, h2, h3, h4 {{
          font-family: 'Orbitron', monospace !important;
          color: {CLR_ACCENT} !important;
          letter-spacing: 0.05em;
      }}

      .metric-card {{
          background: {CLR_CARD};
          border: 1px solid {CLR_BORDER};
          border-left: 3px solid {CLR_ACCENT};
          border-radius: 6px;
          padding: 16px 20px;
          margin: 4px 0;
      }}
      .metric-label {{
          font-family: 'Share Tech Mono', monospace;
          font-size: 11px; color: {CLR_MUTED};
          text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 4px;
      }}
      .metric-value {{
          font-family: 'Orbitron', monospace;
          font-size: 26px; font-weight: 700; color: {CLR_ACCENT}; line-height: 1;
      }}
      .metric-sub {{ font-size: 12px; color: {CLR_MUTED}; margin-top: 4px; }}

      .log-box {{
          background: #080a0f;
          border: 1px solid {CLR_BORDER};
          border-top: 2px solid {CLR_ACCENT};
          border-radius: 4px;
          padding: 14px 16px;
          font-family: 'Share Tech Mono', monospace;
          font-size: 12px; line-height: 1.7; color: #7ecfce;
          max-height: 320px; overflow-y: auto;
          white-space: pre-wrap; word-break: break-all;
      }}
      .log-ppo  {{ color: {CLR_ACCENT}; }}
      .log-ep   {{ color: {CLR_ACCENT3}; }}
      .log-ckpt {{ color: {CLR_ACCENT2}; }}
      .log-err  {{ color: #ff4444; }}

      .media-card {{
          background: {CLR_CARD};
          border: 1px solid {CLR_BORDER};
          border-radius: 8px; padding: 12px; text-align: center;
      }}
      .media-label {{
          font-family: 'Share Tech Mono', monospace;
          font-size: 11px; color: {CLR_MUTED}; margin-top: 8px; text-transform: uppercase;
      }}

      .section-header {{
          font-family: 'Orbitron', monospace;
          font-size: 13px; color: {CLR_MUTED};
          text-transform: uppercase; letter-spacing: 0.2em;
          padding: 18px 0 8px;
          border-bottom: 1px solid {CLR_BORDER}; margin-bottom: 14px;
      }}

      /* Educational cards */
      .edu-card {{
          background: {CLR_CARD};
          border: 1px solid {CLR_BORDER};
          border-radius: 8px; padding: 20px 24px; margin-bottom: 12px;
      }}
      .edu-title {{
          font-family: 'Orbitron', monospace;
          font-size: 14px; color: {CLR_ACCENT}; margin-bottom: 10px;
          letter-spacing: 0.05em;
      }}
      .edu-text {{
          font-family: 'Rajdhani', sans-serif;
          font-size: 15px; line-height: 1.7; color: {CLR_TEXT};
      }}
      .highlight {{ color: {CLR_ACCENT}; font-weight: 600; }}
      .highlight2 {{ color: {CLR_ACCENT2}; font-weight: 600; }}
      .highlight3 {{ color: {CLR_ACCENT3}; font-weight: 600; }}

      .badge {{
          display: inline-block; padding: 2px 10px; border-radius: 20px;
          font-size: 11px; font-family: 'Share Tech Mono', monospace; font-weight: 600;
      }}
      .badge-running {{ background: #0d2e1a; color: {CLR_ACCENT3}; border: 1px solid {CLR_ACCENT3}; }}
      .badge-waiting {{ background: #1e1e0d; color: #ffd700; border: 1px solid #ffd700; }}

      .block-container {{ padding-top: 2rem !important; max-width: 1400px !important; }}
      ::-webkit-scrollbar {{ width: 4px; }}
      ::-webkit-scrollbar-track {{ background: {CLR_BG}; }}
      ::-webkit-scrollbar-thumb {{ background: {CLR_BORDER}; border-radius: 2px; }}

      .stButton > button {{
          background: transparent; border: 1px solid {CLR_ACCENT};
          color: {CLR_ACCENT}; font-family: 'Share Tech Mono', monospace;
          font-size: 12px; letter-spacing: 0.1em; border-radius: 3px;
      }}
      .stButton > button:hover {{ background: rgba(0,229,255,0.08); color: white; }}

      /* Pulse animation for active elements */
      @keyframes pulse-glow {{
          0%, 100% {{ box-shadow: 0 0 4px rgba(0,229,255,0.3); }}
          50%        {{ box-shadow: 0 0 14px rgba(0,229,255,0.7); }}
      }}
      .active-glow {{ animation: pulse-glow 2s infinite; }}
    </style>
    """, unsafe_allow_html=True)

def render_svg(svg_string):
    """Securely renders raw SVG by converting to base64 to avoid Streamlit markdown parser bugs."""
    b64 = base64.b64encode(svg_string.encode('utf-8')).decode('utf-8')
    html = f'<img src="data:image/svg+xml;base64,{b64}" style="width:100%;max-width:760px;display:block;margin:0 auto"/>'
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  PLOTLY BASE LAYOUT
# ═══════════════════════════════════════════════════
BASE_LAYOUT = dict(
    paper_bgcolor = PLOT_BG,
    plot_bgcolor  = PLOT_SURFACE,
    font          = dict(family="Share Tech Mono, monospace", color=CLR_TEXT, size=11),
    margin        = dict(l=50, r=20, t=40, b=40),
    xaxis         = dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, tickfont=dict(size=10)),
    yaxis         = dict(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, tickfont=dict(size=10)),
)


# ═══════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════
@st.cache_data(ttl=5)
def load_training_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Normalize column names while ensuring we don't duplicate names (prevents Series/Array error)
        renames = {}
        seen_targets = set()
        
        for c in df.columns:
            cl = c.strip().lower()
            target = None
            if cl in ("episode", "ep"):           target = "episode"
            elif cl == "reward":                  target = "reward"
            elif cl == "copper":                  target = "copper"
            elif cl == "pol_loss":                target = "pol_loss"
            elif cl == "val_loss":                target = "val_loss"
            elif cl == "entropy":                 target = "entropy"
            elif cl in ("steps", "total_steps"):  target = "steps"
            
            if target and target not in seen_targets:
                renames[c] = target
                seen_targets.add(target)

        df = df.rename(columns=renames)
        
        # Convert numeric columns securely
        for col in ["episode","reward","copper","pol_loss","val_loss","entropy","steps"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                
        df = df.dropna(subset=["episode"] if "episode" in df.columns else [])
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"CSV read error: {e}")
        return pd.DataFrame()


def rolling_avg(series, w=50):
    return series.rolling(window=w, min_periods=1).mean()


def load_log_lines(path="training.log", n=60):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return [l.rstrip() for l in lines[-n:]]
        except Exception:
            pass
    return []


def load_checkpoint_info(path):
    if not os.path.exists(path):
        return {}
    try:
        import torch
        ck = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "episode": ck.get("episode", ck.get("ep", "—")),
            "steps":   ck.get("steps",   ck.get("total_steps", "—")),
            "updates": ck.get("updates", ck.get("total_updates", "—")),
            "best":    ck.get("best",    ck.get("best_reward", "—")),
        }
    except Exception:
        return {}


def find_media(directory):
    exts = {".gif", ".mp4", ".webm", ".jpg", ".jpeg", ".png"}
    p = Path(directory)
    if not p.exists():
        return []
    return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])


# ═══════════════════════════════════════════════════
#  COMPONENTS
# ═══════════════════════════════════════════════════
def metric_card(label, value, sub="", accent=CLR_ACCENT):
    st.markdown(f"""
    <div class="metric-card" style="border-left-color:{accent}">
      <div class="metric-label">{label}</div>
      <div class="metric-value" style="color:{accent}">{value}</div>
      {"<div class='metric-sub'>" + sub + "</div>" if sub else ""}
    </div>""", unsafe_allow_html=True)


def section_header(title):
    st.markdown(f"""
    <div class="section-header">
      <span style="display:inline-block;width:3px;height:16px;
            background:{CLR_ACCENT};border-radius:2px;
            margin-right:10px;vertical-align:middle"></span>
      {title}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  CHARTS
# ═══════════════════════════════════════════════════
def chart_main_metrics(df):
    """Combined subplot chart mimicking the requested layout"""
    if "episode" not in df or "reward" not in df or "copper" not in df:
        return None
        
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.08,
        subplot_titles=("REWARD", "COPPER MINED")
    )
    
    # ── Row 1: Reward ──
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["reward"], 
        mode="lines", line=dict(color=PLOT_ACCENT_DIM, width=1), 
        name="Reward (raw)", showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["episode"], y=rolling_avg(df["reward"], 50), 
        mode="lines", line=dict(color=PLOT_ACCENT, width=2), 
        name="Reward (avg50)"
    ), row=1, col=1)
    
    # ── Row 2: Copper ──
    fig.add_trace(go.Scatter(
        x=df["episode"], y=df["copper"], 
        mode="lines", line=dict(color=PLOT_ACCENT3_DIM, width=1), 
        name="Copper (raw)", showlegend=False
    ), row=2, col=1)
    
    fig.add_trace(go.Scatter(
        x=df["episode"], y=rolling_avg(df["copper"], 50), 
        mode="lines", line=dict(color=PLOT_ACCENT3, width=2), 
        name="Copper (avg50)"
    ), row=2, col=1)

    # Применяем базовые настройки
    fig.update_layout(**BASE_LAYOUT)
    
    # Перезаписываем специфичные настройки вторым вызовом (теперь ошибки не будет)
    fig.update_layout(
        legend=dict(
            orientation="h", yanchor="bottom", y=1.05, xanchor="right", x=1,
            bgcolor=PLOT_BG, bordercolor=PLOT_BG, font=dict(size=10)
        ),
        title=dict(text="MINDUSTRY SIM — PPO (VecEnv)", font=dict(size=14, color=CLR_MUTED)),
        height=550,
        margin=dict(l=50, r=20, t=70, b=40)
    )
    
    # Setup Axes and Grids specific to subplots
    for i in [1, 2]:
        fig.update_xaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, tickfont=dict(size=10), row=i, col=1)
        fig.update_yaxes(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, tickfont=dict(size=10), row=i, col=1)
        
    fig.update_yaxes(title_text="Reward", title_font=dict(size=11), row=1, col=1)
    fig.update_yaxes(title_text="Copper", title_font=dict(size=11), row=2, col=1)
    fig.update_xaxes(title_text="Episode", title_font=dict(size=11), row=2, col=1)
    
    # Restyle subplot titles 
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=11, color=CLR_MUTED, family="Share Tech Mono, monospace")
        
    return fig


def chart_losses(df):
    has_pol = "pol_loss" in df.columns
    has_val = "val_loss" in df.columns
    has_ent = "entropy"  in df.columns
    if not (has_pol or has_val or has_ent):
        return None
    ncols = sum([has_pol, has_val, has_ent])
    titles = []
    if has_pol: titles.append("Policy Loss")
    if has_val: titles.append("Value Loss")
    if has_ent: titles.append("Entropy")

    fig = make_subplots(rows=1, cols=ncols, subplot_titles=titles)
    x   = df.get("episode", pd.Series(range(len(df))))
    col = 1
    if has_pol:
        fig.add_trace(go.Scatter(x=x, y=df["pol_loss"],
            mode="lines", line=dict(color=PLOT_ACCENT, width=1.5), name="pol",
            showlegend=False), row=1, col=col); col += 1
    if has_val:
        fig.add_trace(go.Scatter(x=x, y=df["val_loss"],
            mode="lines", line=dict(color=PLOT_ACCENT2, width=1.5), name="val",
            showlegend=False), row=1, col=col); col += 1
    if has_ent:
        fig.add_trace(go.Scatter(x=x, y=df["entropy"],
            mode="lines", line=dict(color=PLOT_ACCENT3, width=1.5), name="ent",
            showlegend=False), row=1, col=col)

    fig.update_layout(**BASE_LAYOUT,
        title=dict(text="PPO TRAINING LOSSES", font=dict(size=12, color=CLR_MUTED)),
        height=260, showlegend=False,
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, tickfont=dict(size=9))
            
    # Restyle subplot titles 
    for annotation in fig['layout']['annotations']:
        if annotation['text'] in titles:
            annotation['font'] = dict(size=10, color=CLR_MUTED, family="Share Tech Mono, monospace")
            
    return fig


# ═══════════════════════════════════════════════════
#  RL LOOP ANIMATION (SVG)
# ═══════════════════════════════════════════════════
RL_CYCLE_SVG = """
<svg viewBox="0 0 700 220" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <polygon points="0 0, 8 3, 0 6" fill="#00e5ff"/>
    </marker>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Environment -->
  <rect x="20" y="70" width="150" height="80" rx="8"
        fill="#13161e" stroke="#00e5ff" stroke-width="1.5"/>
  <text x="95" y="104" text-anchor="middle" font-family="Orbitron,monospace"
        font-size="11" fill="#00e5ff" filter="url(#glow)">ENVIRONMENT</text>
  <text x="95" y="122" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">Mindustry</text>
  <text x="95" y="137" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">WebSocket</text>

  <!-- Agent -->
  <rect x="275" y="70" width="150" height="80" rx="8"
        fill="#13161e" stroke="#7fff6b" stroke-width="1.5"/>
  <text x="350" y="104" text-anchor="middle" font-family="Orbitron,monospace"
        font-size="11" fill="#7fff6b" filter="url(#glow)">AGENT</text>
  <text x="350" y="122" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">PPO + LSTM</text>
  <text x="350" y="137" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">CNN + MLP</text>

  <!-- Memory -->
  <rect x="530" y="70" width="150" height="80" rx="8"
        fill="#13161e" stroke="#ff6b35" stroke-width="1.5"/>
  <text x="605" y="104" text-anchor="middle" font-family="Orbitron,monospace"
        font-size="11" fill="#ff6b35" filter="url(#glow)">MEMORY</text>
  <text x="605" y="122" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">Rollout Buffer</text>
  <text x="605" y="137" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">GAE + PPO</text>

  <!-- Top Arrows: environment->agent (observation + reward) -->
  <line x1="172" y1="95" x2="273" y2="95"
        stroke="#00e5ff" stroke-width="1.5" marker-end="url(#arr)">
    <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="2s" repeatCount="indefinite"/>
  </line>
  <text x="222" y="88" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#00e5ff">obs + reward</text>

  <!-- Arrow: agent->environment (action) -->
  <line x1="273" y1="135" x2="172" y2="135"
        stroke="#7fff6b" stroke-width="1.5" marker-end="url(#arr)">
    <animate attributeName="stroke-opacity" values="1;0.3;1" dur="2s" repeatCount="indefinite"/>
  </line>
  <text x="222" y="150" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#7fff6b">action</text>

  <!-- Arrow: agent->memory (save transitions) -->
  <line x1="427" y1="95" x2="528" y2="95"
        stroke="#ff6b35" stroke-width="1.5" marker-end="url(#arr)">
    <animate attributeName="stroke-opacity" values="0.3;1;0.3" dur="3s" repeatCount="indefinite"/>
  </line>
  <text x="477" y="88" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#ff6b35">transitions</text>

  <!-- Arrow: memory->agent (weight update) -->
  <line x1="528" y1="135" x2="427" y2="135"
        stroke="#ffd700" stroke-width="1.5" marker-end="url(#arr)">
    <animate attributeName="stroke-opacity" values="1;0.3;1" dur="3s" repeatCount="indefinite"/>
  </line>
  <text x="477" y="150" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="8" fill="#ffd700">gradient update</text>

  <!-- Bottom Label -->
  <text x="350" y="200" text-anchor="middle" font-family="Share Tech Mono,monospace"
        font-size="9" fill="#4a5568">
    REINFORCEMENT LEARNING LOOP  //  1024 steps per rollout  //  PPO update every rollout
  </text>
</svg>
"""


# ═══════════════════════════════════════════════════
#  NEURAL NETWORK ARCHITECTURE ANIMATION (SVG)
# ═══════════════════════════════════════════════════
NN_ARCH_SVG = """
<svg viewBox="0 0 760 280" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arr2" markerWidth="7" markerHeight="5" refX="7" refY="2.5" orient="auto">
      <polygon points="0 0, 7 2.5, 0 5" fill="#4a5568"/>
    </marker>
    <filter id="glow2">
      <feGaussianBlur stdDeviation="2" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- MAP INPUT -->
  <rect x="10" y="100" width="70" height="80" rx="4" fill="#0d1520" stroke="#1e2330" stroke-width="1"/>
  <text x="45" y="135" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#4a5568">MAP</text>
  <text x="45" y="148" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#4a5568">32×32</text>
  <text x="45" y="161" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#4a5568">×11ch</text>

  <!-- SCALARS INPUT -->
  <rect x="10" y="195" width="70" height="40" rx="4" fill="#0d1520" stroke="#1e2330" stroke-width="1"/>
  <text x="45" y="213" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#4a5568">SCALARS</text>
  <text x="45" y="226" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#4a5568">16-dim</text>

  <!-- CNN -->
  <rect x="110" y="80" width="90" height="120" rx="6" fill="#13161e" stroke="#00e5ff" stroke-width="1.5"/>
  <text x="155" y="108" text-anchor="middle" font-family="Orbitron" font-size="9" fill="#00e5ff" filter="url(#glow2)">CNN</text>
  <text x="155" y="126" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">Conv 32</text>
  <text x="155" y="140" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">ResBlock</text>
  <text x="155" y="154" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">Conv 64</text>
  <text x="155" y="168" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">ResBlock</text>
  <text x="155" y="182" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">→ 4096</text>

  <!-- MLP -->
  <rect x="110" y="195" width="90" height="55" rx="6" fill="#13161e" stroke="#ff6b35" stroke-width="1.5"/>
  <text x="155" y="218" text-anchor="middle" font-family="Orbitron" font-size="9" fill="#ff6b35">MLP</text>
  <text x="155" y="234" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">Linear 128</text>
  <text x="155" y="245" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">→ 128</text>

  <!-- LSTM -->
  <rect x="240" y="100" width="100" height="100" rx="6" fill="#13161e" stroke="#ffd700" stroke-width="1.5"/>
  <text x="290" y="130" text-anchor="middle" font-family="Orbitron" font-size="10" fill="#ffd700" filter="url(#glow2)">LSTM</text>
  <text x="290" y="150" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">hidden 256</text>
  <text x="290" y="164" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">i,f,g,o gates</text>
  <text x="290" y="178" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">memory state</text>
  <!-- Loop arrow LSTM -->
  <path d="M 340 130 Q 380 110 380 150 Q 380 190 340 180"
        stroke="#ffd700" stroke-width="1" fill="none" stroke-dasharray="3,2" opacity="0.5"/>
  <text x="388" y="155" font-family="Share Tech Mono" font-size="7" fill="#ffd700" opacity="0.7">h,c</text>

  <!-- ACTOR HEAD -->
  <rect x="420" y="60" width="110" height="120" rx="6" fill="#13161e" stroke="#7fff6b" stroke-width="1.5"/>
  <text x="475" y="85" text-anchor="middle" font-family="Orbitron" font-size="9" fill="#7fff6b" filter="url(#glow2)">ACTOR</text>
  <text x="475" y="105" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">type (4)</text>
  <text x="475" y="119" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">↓ embed</text>
  <text x="475" y="133" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">x (32)</text>
  <text x="475" y="147" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">↓ embed</text>
  <text x="475" y="161" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">y (32)</text>
  <text x="475" y="175" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">move (2)</text>

  <!-- CRITIC HEAD -->
  <rect x="420" y="195" width="110" height="60" rx="6" fill="#13161e" stroke="#ff6b35" stroke-width="1.5"/>
  <text x="475" y="218" text-anchor="middle" font-family="Orbitron" font-size="9" fill="#ff6b35">CRITIC</text>
  <text x="475" y="235" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">256 → 128 → 1</text>
  <text x="475" y="249" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">V(state)</text>

  <!-- OUTPUTS -->
  <rect x="575" y="60" width="90" height="50" rx="4" fill="#0d2010" stroke="#7fff6b" stroke-width="1"/>
  <text x="620" y="82" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#7fff6b">ACTION</text>
  <text x="620" y="97" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">type,x,y,vx,vy</text>

  <rect x="575" y="120" width="90" height="40" rx="4" fill="#0d2010" stroke="#7fff6b" stroke-width="1"/>
  <text x="620" y="138" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#7fff6b">LOG_PROB</text>
  <text x="620" y="152" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">for PPO loss</text>

  <rect x="575" y="195" width="90" height="40" rx="4" fill="#1a0d08" stroke="#ff6b35" stroke-width="1"/>
  <text x="620" y="213" text-anchor="middle" font-family="Share Tech Mono" font-size="8" fill="#ff6b35">VALUE</text>
  <text x="620" y="227" text-anchor="middle" font-family="Share Tech Mono" font-size="7" fill="#4a5568">V(s) baseline</text>

  <!-- Connection Arrows -->
  <line x1="82" y1="140" x2="108" y2="140" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="82" y1="215" x2="108" y2="215" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="202" y1="140" x2="238" y2="150" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="202" y1="222" x2="238" y2="180" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="342" y1="130" x2="418" y2="100" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="342" y1="170" x2="418" y2="220" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="532" y1="100" x2="573" y2="85" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="532" y1="120" x2="573" y2="135" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>
  <line x1="532" y1="220" x2="573" y2="215" stroke="#4a5568" stroke-width="1" marker-end="url(#arr2)"/>

  <!-- Signature -->
  <text x="380" y="270" text-anchor="middle" font-family="Share Tech Mono" font-size="9" fill="#4a5568">
    ACTOR-CRITIC ARCHITECTURE  //  ~2.5M parameters
  </text>
</svg>
"""


# ═══════════════════════════════════════════════════
#  PPO ANIMATION (Interactive Plotly)
# ═══════════════════════════════════════════════════
def ppo_concept_chart():
    """Shows PPO clipping — why clip_eps is needed."""
    import numpy as np
    r = np.linspace(0, 2.5, 300)
    adv = 1.0
    eps = 0.2
    obj_normal  = r * adv
    obj_clipped = np.clip(r, 1 - eps, 1 + eps) * adv
    ppo_obj     = np.minimum(obj_normal, obj_clipped)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=r, y=obj_normal, mode="lines",
        line=dict(color="rgba(255,107,53,0.4)", width=1.5, dash="dot"),
        name="No clipping (dangerous)"))
    fig.add_trace(go.Scatter(x=r, y=obj_clipped, mode="lines",
        line=dict(color="rgba(0,229,255,0.4)", width=1.5, dash="dash"),
        name="Clipped"))
    fig.add_trace(go.Scatter(x=r, y=ppo_obj, mode="lines",
        line=dict(color=PLOT_ACCENT, width=2.5),
        name="PPO objective (min)"))

    fig.add_vline(x=1-eps, line=dict(color="rgba(255,215,0,0.5)", width=1, dash="dot"))
    fig.add_vline(x=1+eps, line=dict(color="rgba(255,215,0,0.5)", width=1, dash="dot"))
    fig.add_vrect(x0=1-eps, x1=1+eps, fillcolor="rgba(255,215,0,0.05)", line_width=0)

    fig.add_annotation(x=1.0, y=1.15, text=f"clip [{1-eps}, {1+eps}]",
        showarrow=False, font=dict(family="Share Tech Mono", size=10, color="#ffd700"))

    fig.update_layout(**BASE_LAYOUT,
        legend=dict(orientation="h", y=-0.25, bgcolor=CLR_CARD,
                    bordercolor=CLR_BORDER, borderwidth=1, font=dict(size=10)),
        title=dict(text="PPO CLIPPING — WHY WE LIMIT THE UPDATE STEP",
                   font=dict(size=12, color=CLR_MUTED)),
        xaxis_title="ratio = pi_new / pi_old",
        yaxis_title="objective (advantage=1)",
        height=300,
    )
    return fig


def reward_shaping_chart():
    """Shows how the reward is structured in our agent."""
    import numpy as np
    dist = np.linspace(0, 1, 200)

    r_proximity   = 1.0 * (1 - dist)
    r_approach    = np.where(dist < 0.5, 3.0 * (0.5 - dist), 0)
    r_total       = r_proximity + 0.5

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dist, y=r_proximity, mode="lines",
        line=dict(color=PLOT_ACCENT, width=2), name="r_near_base x (1 - dist)"))
    fig.add_trace(go.Scatter(x=dist, y=r_total, mode="lines",
        line=dict(color=PLOT_ACCENT3, width=2, dash="dash"), name="Total reward (schema)"))
    fig.add_vline(x=0.0, line=dict(color="rgba(127,255,107,0.5)", width=1))
    fig.add_annotation(x=0.05, y=1.4, text="Base", showarrow=False,
        font=dict(family="Share Tech Mono", size=9, color=PLOT_ACCENT3))
    fig.add_annotation(x=0.95, y=0.1, text="Map edge", showarrow=False,
        font=dict(family="Share Tech Mono", size=9, color=CLR_MUTED))

    fig.update_layout(**BASE_LAYOUT,
        legend=dict(orientation="h", y=-0.25, bgcolor=CLR_CARD,
                    bordercolor=CLR_BORDER, borderwidth=1, font=dict(size=10)),
        title=dict(text="REWARD SHAPING — PROXIMITY TO BASE",
                   font=dict(size=12, color=CLR_MUTED)),
        xaxis_title="Distance to base (norm.)",
        yaxis_title="Reward",
        height=280,
    )
    return fig


# ═══════════════════════════════════════════════════
#  LOGS
# ═══════════════════════════════════════════════════
def colorize_log(lines):
    out = []
    for line in lines:
        if "[PPO]" in line:
            out.append(f'<span class="log-ppo">{line}</span>')
        elif "[Ep " in line or "reward" in line.lower():
            out.append(f'<span class="log-ep">{line}</span>')
        elif "[Ckpt]" in line or "[Checkpoint]" in line:
            out.append(f'<span class="log-ckpt">{line}</span>')
        elif "error" in line.lower() or "exception" in line.lower():
            out.append(f'<span class="log-err">{line}</span>')
        else:
            out.append(line)
    return "\n".join(out)


# ═══════════════════════════════════════════════════
#  MAIN PAGE
# ═══════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="MIND-RL // Mindustry Agent",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()

    # ── Header ─────────────────────────────────
    st.markdown("""
    <h1 style="font-size:32px;margin-bottom:0;line-height:1">MIND-RL</h1>
    <div style="font-family:'Share Tech Mono',monospace;font-size:13px;
                color:#4a5568;margin-top:2px;letter-spacing:0.15em">
      MINDUSTRY REINFORCEMENT LEARNING AGENT // PPO + LSTM
    </div>
    <hr style="border-color:#1e2330;margin:10px 0 20px">
    """, unsafe_allow_html=True)

    col_ref, col_badge, _ = st.columns([1, 1, 5])
    with col_ref:
        auto = st.toggle("Auto-update", value=False)
    with col_badge:
        exists = os.path.exists(CKPT_PATH)
        cls  = "badge-running" if exists else "badge-waiting"
        text = "TRAINED" if exists else "NO CHECKPOINT"
        st.markdown(f'<div style="padding-top:6px"><span class="badge {cls}">{text}</span></div>',
                    unsafe_allow_html=True)
    if auto:
        time.sleep(5); st.rerun()

    # ══════════════════════════════════════════════
    #  ABOUT PROJECT
    # ══════════════════════════════════════════════
    section_header("ABOUT THE PROJECT")
    st.markdown(f"""
    <div class="edu-card">
      <div class="edu-text">
        <span class="highlight">MIND-RL</span> — an agent based on
        <span class="highlight">Proximal Policy Optimization (PPO)</span> with
        <span class="highlight">LSTM</span>, trained to play
        <span class="highlight2">Mindustry</span>.
        The agent connects to the game via a <b>WebSocket mod</b>, receives a 32x32 map (11 channels)
        and 16 scalar features, and selects: action type (idle/drill/conveyor/delete),
        building coordinates, and movement vector.
        <br><br>
        <b>Goal:</b> explore the map → find copper → build a drill → build a conveyor → deliver to base.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    #  EDUCATIONAL SECTION
    # ══════════════════════════════════════════════
    section_header("HOW IT WORKS")

    tab1, tab2, tab3, tab4 = st.tabs([
        "What is RL",
        "Model Architecture",
        "PPO Algorithm",
        "Reward Shaping",
    ])

    # ── Tab 1: What is RL ──────────────────────
    with tab1:
        st.markdown(f"""
        <div class="edu-card">
          <div class="edu-title">REINFORCEMENT LEARNING — LEARNING THROUGH EXPERIENCE</div>
          <div class="edu-text">
            Unlike conventional machine learning, where a model learns from pre-existing data,
            an RL agent <span class="highlight">collects its own experience</span> by interacting with the environment.
            <br><br>
            The idea is simple: the agent takes an action → the environment responds with an observation and a reward →
            the agent updates its strategy to get <span class="highlight">more reward in the future</span>.
            No labeled data, no teacher — just trial and error.
          </div>
        </div>
        """, unsafe_allow_html=True)

        render_svg(RL_CYCLE_SVG)

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"""
            <div class="edu-card" style="border-left:3px solid {CLR_ACCENT}">
              <div class="edu-title" style="font-size:12px">OBSERVATION (obs)</div>
              <div class="edu-text" style="font-size:13px">
                Map around the agent 32x32 tiles,
                11 channels (ground, copper, conveyors, drills, enemies...) +
                16 scalars (health, position, distance to base, etc.)
              </div>
            </div>""", unsafe_allow_html=True)
        with col_b:
            st.markdown(f"""
            <div class="edu-card" style="border-left:3px solid {CLR_ACCENT3}">
              <div class="edu-title" style="font-size:12px">ACTION (action)</div>
              <div class="edu-text" style="font-size:13px">
                Type (idle / drill / conveyor / delete) + map coordinates (x, y) +
                continuous movement vector (vx, vy) — totaling ~4100 combinations.
              </div>
            </div>""", unsafe_allow_html=True)
        with col_c:
            st.markdown(f"""
            <div class="edu-card" style="border-left:3px solid {CLR_ACCENT2}">
              <div class="edu-title" style="font-size:12px">REWARD (reward)</div>
              <div class="edu-text" style="font-size:13px">
                Proximity to base x constantly + bonus for approaching base +
                penalty for standing still + bonus for mining copper.
                Normalized using a running standard deviation.
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Tab 2: Architecture ───────────────────────
    with tab2:
        st.markdown(f"""
        <div class="edu-card">
          <div class="edu-title">ACTOR-CRITIC + LSTM</div>
          <div class="edu-text">
            The neural network is split into two parts:
            <span class="highlight">Actor</span> (chooses the action) and
            <span class="highlight2">Critic</span> (evaluates how good the current state is).
            <br><br>
            Map data is processed by a
            <span class="highlight">CNN with ResBlocks</span> — just like in computer vision.
            Scalars are processed by a separate
            <span class="highlight">MLP</span>.
            Both streams merge into an
            <span class="highlight" style="color:#ffd700">LSTM</span> — a recurrent network
            that remembers what happened in <i>previous</i> steps.
            This is crucial: the agent only sees a piece of the map, and without memory, it wouldn't know where it came from.
          </div>
        </div>
        """, unsafe_allow_html=True)

        render_svg(NN_ARCH_SVG)

        st.markdown(f"""
        <div class="edu-card" style="margin-top:12px">
          <div class="edu-title">HIERARCHICAL ACTION</div>
          <div class="edu-text">
            The actor chooses an action <span class="highlight">sequentially</span>:
            first the type (4 options), then X (32 options), then Y (32 options).
            This is much more efficient than trying to guess all 4096 combinations at once.
            <br><br>
            Movement (vx, vy) is <span class="highlight2">continuous</span>, drawn from a normal distribution.
            This allows the agent to move smoothly instead of jumping across tiles.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 3: PPO ───────────────────────────────
    with tab3:
        st.markdown(f"""
        <div class="edu-card">
          <div class="edu-title">PROXIMAL POLICY OPTIMIZATION (PPO)</div>
          <div class="edu-text">
            The main problem in RL training — if the weights are updated too much,
            the agent will "forget" what worked and start behaving unpredictably.
            <br><br>
            PPO solves this by <span class="highlight">clipping the ratio</span> of the new and old policy.
            If the new policy differs from the old one by more than ε=0.2 —
            the gradient simply won't pass. This is a <span class="highlight">safe update step</span>.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(ppo_concept_chart(), use_container_width=True,
                        config={"displayModeBar": False})

        col_p, col_v = st.columns(2)
        with col_p:
            st.markdown(f"""
            <div class="edu-card">
              <div class="edu-title" style="font-size:12px">GAE — ADVANTAGE ESTIMATION</div>
              <div class="edu-text" style="font-size:13px">
                Instead of a simple reward, we use <span class="highlight">Generalized Advantage Estimation</span> —
                a weighted sum of future rewards with discount γ=0.99 and λ=0.95.
                This reduces the variance of updates and speeds up training.
              </div>
            </div>""", unsafe_allow_html=True)
        with col_v:
            st.markdown(f"""
            <div class="edu-card">
              <div class="edu-title" style="font-size:12px">ENTROPY BONUS</div>
              <div class="edu-text" style="font-size:13px">
                An <span class="highlight2">entropy bonus</span> (0.03) is added to the loss function.
                This prevents the agent from "locking in" on a single action too early
                and encourages it to keep exploring the map even after a few successful attempts.
              </div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="edu-card" style="margin-top:4px">
          <div class="edu-title" style="font-size:12px">UPDATE CYCLE</div>
          <div class="edu-text" style="font-size:14px">
            <span class="highlight">512 env steps</span> (rollout)
            → calculate GAE advantages
            → shuffle data
            → <span class="highlight">3 passes</span> over batches of 64 transitions
            → update weights
            → repeat.
            <br>
            The backward pass is executed on the <span class="highlight2">CPU</span>
            (DirectML doesn't support scatter operations),
            while inference is on the <span class="highlight">GPU via DirectML</span>.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tab 4: Reward Shaping ────────────────────
    with tab4:
        st.markdown(f"""
        <div class="edu-card">
          <div class="edu-title">REWARD SHAPING — HOW THE AGENT "FEELS" THE RULES</div>
          <div class="edu-text">
            The agent doesn't know the game rules. It only knows one thing:
            <span class="highlight">the higher the total reward, the better</span>.
            Therefore, reward shaping is a way to "explain" to the agent what to do
            through the numbers it receives every step.
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(reward_shaping_chart(), use_container_width=True,
                        config={"displayModeBar": False})

        components = [
            (CLR_ACCENT,  "r_near_base × (1 - dist)",   "+1.0/step", "Constant bonus for base proximity. The agent ALWAYS wants to be near the base."),
            (CLR_ACCENT3, "r_approach × delta",         "+3.0×Δd",   "Bonus for each step closer to the base. Motivates it to move towards it."),
            (CLR_ACCENT2, "r_leave × |delta|",          "−2.0×Δd",   "Penalty for moving away from the base. Prevents aimless running around the map."),
            ("#ffd700",   "r_no_move × over_grace",      "−0.5/step", "Linear penalty for standing still for more than 10 steps."),
            ("#cc88ff",   "copper × r_copper_mult",      "+5.0×cop",  "Secondary signal. Copper is important, but not the main priority — find the base first."),
        ]
        for accent, formula, value, desc in components:
            st.markdown(f"""
            <div class="edu-card" style="border-left:3px solid {accent};padding:12px 20px;margin-bottom:8px">
              <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:4px">
                <code style="font-family:'Share Tech Mono';font-size:12px;color:{accent}">{formula}</code>
                <span style="font-family:'Orbitron';font-size:12px;color:{accent}">{value}</span>
              </div>
              <div class="edu-text" style="font-size:13px">{desc}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="edu-card" style="border-left:3px solid #4a5568;margin-top:8px">
          <div class="edu-title" style="font-size:12px;color:{CLR_MUTED}">REWARD NORMALIZATION (Welford)</div>
          <div class="edu-text" style="font-size:13px">
            All rewards are divided by a <span class="highlight">running standard deviation</span>
            (Welford's algorithm, online computation). This ensures the val_loss
            stays in the 0.5–5 range, rather than exploding to 900+ as it did before.
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════
    #  STATUS AND METRICS
    # ══════════════════════════════════════════════
    section_header("TRAINING STATUS")
    ck = load_checkpoint_info(CKPT_PATH)
    df = load_training_data(LOG_CSV)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        ep_v = ck.get("episode", len(df) if not df.empty else "—")
        metric_card("Episodes", str(ep_v), "total")
    with c2:
        sv = ck.get("steps", "—")
        sf = f"{sv/1000:.1f}K" if isinstance(sv, (int, float)) else str(sv)
        metric_card("Steps", sf, "env steps", CLR_ACCENT2)
    with c3:
        metric_card("PPO Updates", str(ck.get("updates", "—")), "gradient updates", CLR_ACCENT3)
    with c4:
        bv = ck.get("best", "—")
        bf = f"{bv:.3f}" if isinstance(bv, float) else str(bv)
        metric_card("Best Reward", bf, "normalized", "#ff9f43")
    with c5:
        cv = "—"
        if not df.empty and "copper" in df.columns:
            cv = f"{df['copper'].sum():.0f}"
        metric_card("Copper (total)", cv, "all episodes", CLR_ACCENT3)

    # ══════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════
    section_header("TRAINING CHARTS")

    if df.empty:
        st.markdown(f"""
        <div style="background:{CLR_CARD};border:1px solid {CLR_BORDER};
                    border-left:3px solid #ffd700;padding:16px 20px;border-radius:4px;
                    font-family:'Share Tech Mono',monospace;font-size:12px;color:#ffd700">
          File <b>{LOG_CSV}</b> not found or empty. Start training.<br>
          Expected columns: episode, reward, copper, pol_loss, val_loss, entropy
        </div>""", unsafe_allow_html=True)
    else:
        if "episode" in df.columns and len(df) > 1:
            ep_min, ep_max = int(df["episode"].min()), int(df["episode"].max())
            if ep_min < ep_max:
                rng = st.slider("Episode range", ep_min, ep_max, (ep_min, ep_max))
                df_view = df[(df["episode"] >= rng[0]) & (df["episode"] <= rng[1])]
            else:
                df_view = df
        else:
            df_view = df

        fig_main = chart_main_metrics(df_view)
        if fig_main:
            st.plotly_chart(fig_main, use_container_width=True, config={"displayModeBar": False})

        fig_l = chart_losses(df_view)
        if fig_l:
            st.plotly_chart(fig_l, use_container_width=True, config={"displayModeBar": False})

    # ══════════════════════════════════════════════
    #  CONSOLE
    # ══════════════════════════════════════════════
    section_header("CONSOLE")

    log_col, ctrl_col = st.columns([5, 1])
    with ctrl_col:
        n_lines    = st.number_input("Lines", 10, 200, 50, 10)
        log_path   = st.text_input("Log file", "training.log")
    with log_col:
        lines = load_log_lines(log_path, int(n_lines))
        if lines:
            st.markdown(f'<div class="log-box">{colorize_log(lines)}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="log-box" style="color:{CLR_MUTED}">
Log not found: {log_path}

Add file logging to your agent:
  import logging
  logging.basicConfig(filename='training.log', level=logging.INFO)
  &#35; then use logging.info() instead of print()
            </div>""", unsafe_allow_html=True)

    if not df.empty:
        with st.expander("Latest CSV records"):
            st.dataframe(df.tail(20), use_container_width=True)

    # ══════════════════════════════════════════════
    #  MEDIA GALLERY
    # ══════════════════════════════════════════════
    section_header("DEMO — HOW THE AGENT PLAYS")

    media_dir = st.text_input("Media folder", MEDIA_DIR, key="mdir")
    files     = find_media(media_dir)

    if not files:
        st.markdown(f"""
        <div style="background:{CLR_CARD};border:1px dashed {CLR_BORDER};
                    border-radius:8px;padding:40px;text-align:center">
          <div style="font-family:'Orbitron',monospace;font-size:18px;color:{CLR_MUTED}">
            [ NO MEDIA ]
          </div>
          <div style="font-family:'Share Tech Mono',monospace;font-size:11px;
                      color:{CLR_MUTED};margin-top:8px">
            Create folder <b>{media_dir}/</b> and put .gif, .mp4, .png files there
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        cols = st.columns(min(3, len(files)))
        for i, f in enumerate(files):
            with cols[i % len(cols)]:
                ext = f.suffix.lower()
                st.markdown('<div class="media-card">', unsafe_allow_html=True)
                if ext == ".gif": st.image(str(f), use_container_width=True)
                elif ext in (".mp4", ".webm"): st.video(str(f))
                elif ext in (".jpg", ".jpeg", ".png"): st.image(str(f), use_container_width=True)
                st.markdown(f'<div class="media-label">{f.stem}</div></div>',
                            unsafe_allow_html=True)

    section_header("UPLOAD MEDIA FILE")
    uploaded = st.file_uploader("gif / mp4 / png / jpg",
        type=["gif","mp4","webm","jpg","jpeg","png"], accept_multiple_files=True)
    if uploaded:
        os.makedirs(media_dir, exist_ok=True)
        for uf in uploaded:
            with open(Path(media_dir) / uf.name, "wb") as out:
                out.write(uf.read())
        st.success(f"Saved {len(uploaded)} file(s)"); st.rerun()

    # ── Footer ─────────────────────────────────────
    st.markdown(f"""
    <div style="margin-top:40px;padding:16px 0;border-top:1px solid {CLR_BORDER};
                font-family:'Share Tech Mono',monospace;font-size:10px;color:{CLR_MUTED};
                display:flex;justify-content:space-between">
      <span>MIND-RL // PPO+LSTM // Mindustry Agent</span>
      <span>AMD RX6550M + Ryzen 5 7535HS // DirectML</span>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()