"""
Production-Ready Data Analytics AI Agent
=========================================
A comprehensive data analytics platform powered by Hugging Face LLMs.
"""

import streamlit as st
from utils.session import init_session_state
from utils.config import load_config

# --- Page Configuration ---
st.set_page_config(
    page_title="Data Analyst Agent",
    page_icon="📊",
    layout="wide",  
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/Om-Suman/Data_Analyst_Agent',
        'Report a bug': 'https://github.com/Om-Suman/Data_Analyst_Agent/issues',
        'About': '# DataAgent Pro\nProduction-grade AI-powered data analytics platform.'
    }
)

# Load config and init session
load_config()
init_session_state()

# --- Theme CSS ---
def inject_css():
    
    theme = st.session_state.get("theme", "dark")
    if theme == "dark":
        bg = "#0e1117"
        card_bg = "#1a1d24"
        border = "#2d3748"
        text = "#e2e8f0"
        accent = "#4f8ef7"
        muted = "#718096"
    else:
        bg = "#f7f9fc"
        card_bg = "#ffffff"
        border = "#e2e8f0"
        text = "#1a202c"
        accent = "#3b82f6"
        muted = "#718096"

    st.markdown(f"""
    <style>

    /* Root */
    :root {{
        --bg: {bg};
        --card-bg: {card_bg};
        --border: {border};
        --text: {text};
        --accent: {accent};
        --muted: {muted};
    }}
    .stApp {{ background-color: var(--bg); color: var(--text); }}
    .main .block-container {{ padding: 1.5rem 2rem; max-width: 1400px; }}

    /* Sidebar */
    
    [data-testid="stSidebar"] {{
        background-color: var(--card-bg) !important;
        border-right: 1px solid var(--border);
    }}
    [data-testid="stSidebar"] .stButton button {{
        width: 100%;
        text-align: left;
        background: transparent;
        border: 1px solid var(--border);
        color: var(--text);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        margin: 2px 0;
        transition: all 0.2s;
    }}
    [data-testid="stSidebar"] .stButton button:hover {{
        background: var(--accent) !important;
        border-color: var(--accent) !important;
        color: white !important;
    }}

    /* Cards */
    .metric-card {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.25rem;
        margin: 0.5rem 0;
    }}
    .metric-card h3 {{ margin: 0; font-size: 1.75rem; color: var(--accent); }}
    .metric-card p {{ margin: 0.25rem 0 0; color: var(--muted); font-size: 0.85rem; }}

    /* Insight cards */
    .insight-card {{
        background: var(--card-bg);
        border-left: 4px solid var(--accent);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin: 0.5rem 0;
    }}

    /* Tables */
    .dataframe {{ border-radius: 8px; overflow: hidden; }}

    /* Chat bubbles */
    .chat-user {{
        background: var(--accent);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }}
    .chat-agent {{
        background: var(--card-bg);
        border: 1px solid var(--border);
        border-radius: 18px 18px 18px 4px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        max-width: 85%;
    }}

    /* Buttons */
    .stButton button[kind="primary"] {{
        background: linear-gradient(135deg, {accent}, #7c3aed);
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
    }}

    /* Status badges */
    .badge-success {{ background:#10b981; color:white; border-radius:12px; padding:2px 10px; font-size:0.75rem; }}
    .badge-warning {{ background:#f59e0b; color:white; border-radius:12px; padding:2px 10px; font-size:0.75rem; }}
    .badge-error   {{ background:#ef4444; color:white; border-radius:12px; padding:2px 10px; font-size:0.75rem; }}

    /* Hide streamlit branding */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# --- Navigation ---
from utils.navigation import render_navigation
render_navigation()
