"""
frontend/styles.py — All CSS for the AI Data Scientist dashboard.
Sophisticated Dark Premium Theme.
"""

MAIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif; }

/* ── Background ── */
.stApp {
    background: radial-gradient(circle at 50% 0%, #2d2a6e 0%, #1a2540 100%);
    min-height: 100vh;
    color: #f1f5f9;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #1a2540 !important;
    border-right: 1px solid #2a3d5c !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
    color: #94a3b8;
}

/* ── Hero ── */
.hero {
    background: rgba(42, 58, 90, 0.45);
    backdrop-filter: blur(8px);
    border: 1px solid #3e5070;
    box-shadow: 0 10px 40px -10px rgba(0,0,0,0.4);
    border-radius: 28px;
    padding: 3rem;
    text-align: center;
    margin-bottom: 2rem;
}
.hero h1 {
    font-size: 2.8rem; font-weight: 800;
    background: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin: 0 0 0.8rem; line-height: 1.1;
}
.hero p { color: #94a3b8; font-size: 1.1rem; font-weight: 500; margin: 0; }
.hero .pill-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: #3d3a9e; border: 1px solid #5552c8;
    border-radius: 999px; padding: 0.4rem 1.1rem;
    font-size: 0.8rem; font-weight: 700; color: #c7d2fe;
    margin: 1rem 0.4rem 0;
}

/* ── Section Headers ── */
.sec-hdr {
    font-size: 0.78rem; font-weight: 800; letter-spacing: 0.15em;
    text-transform: uppercase; color: #818cf8;
    margin-bottom: 1rem;
    padding-left: 0.5rem;
}

/* ── Stat Cards ── */
.scard, .dp-stat {
    background: #233050;
    border: 1px solid #3e5070;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    border-radius: 20px; padding: 1.5rem;
    text-align: center; transition: all 0.25s ease;
}
.scard:hover, .dp-stat:hover { 
    border-color: #6366f1; 
    transform: translateY(-4px);
    box-shadow: 0 12px 20px -5px rgba(99, 102, 241, 0.3);
}
.scard .v, .dp-stat .val { font-size: 2rem; font-weight: 800; display: block; color: #ffffff; }
.scard .l, .dp-stat .lbl {
    font-size: 0.75rem; text-transform: uppercase; font-weight: 800;
    letter-spacing: 0.08em; color: #94a3b8; margin-top: 0.4rem;
}

/* ── Insight Box ── */
.insight {
    background: #233050;
    border: 1px solid #3e5070;
    border-left: 6px solid #6366f1;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    color: #e2e8f0;
    font-size: 1rem; line-height: 1.65;
    margin-bottom: 2rem;
}

/* ── Section Banner ── */
.sec-banner {
    border-radius: 20px; padding: 1.4rem 2rem; margin-bottom: 2rem;
    display: flex; align-items: center; gap: 1.5rem;
    background: #233050 !important;
    border: 1px solid #3e5070 !important;
}
.sec-banner .sec-num {
    font-size: 0.75rem; font-weight: 800; letter-spacing: 0.15em;
    text-transform: uppercase; color: #818cf8; margin-bottom: 0.4rem;
}
.sec-banner .sec-title { font-size: 1.3rem; font-weight: 800; color: #ffffff; }

/* ── Card Header ── */
.dp-card-hdr {
    display: flex; align-items: center; gap: 0.8rem;
    padding: 1.1rem 1.5rem;
    background: #1a2540;
    border: 1px solid #2a3d5c;
    border-radius: 14px;
    font-size: 0.95rem; font-weight: 700; color: #c7d2fe;
    margin-bottom: 1rem;
}

/* ── Column Badges ── */
.col-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    border-radius: 12px; padding: 0.5rem 1rem;
    font-size: 0.85rem; font-weight: 700; margin: 0.3rem;
    border: 1px solid transparent;
}
.col-badge.safe { background: rgba(34, 197, 94, 0.15); color: #4ade80; border-color: rgba(34, 197, 94, 0.3); }
.col-badge.warn { background: rgba(234, 179, 8, 0.15); color: #facc15; border-color: rgba(234, 179, 8, 0.3); }
.col-badge.drop { background: rgba(239, 68, 68, 0.15); color: #f87171; border-color: rgba(239, 68, 68, 0.3); }
.col-badge .dr  { font-size: 0.75rem; color: #64748b; font-weight: 500; }

/* ── Tables ── */
.otable, .stattbl {
    width: 100%; border-collapse: separate; border-spacing: 0;
    font-size: 0.95rem; border-radius: 20px; overflow: hidden;
    border: 1px solid #3e5070;
    margin-bottom: 2rem;
}
.otable th, .stattbl th {
    background: #1a2540; color: #94a3b8; font-weight: 800;
    padding: 1.1rem 1.5rem; text-align: left;
    border-bottom: 1px solid #3e5070;
}
.otable td, .stattbl td {
    padding: 1rem 1.5rem;
    border-bottom: 1px solid #2a3d5c;
    color: #e2e8f0;
    background: #233050;
}
.otable tr:hover td, .stattbl tr:hover td { background: #2e4060; }

/* ── Tabs ── */
button[data-baseweb="tab"] {
    font-size: 1rem !important; font-weight: 800 !important;
    color: #94a3b8 !important;
    padding: 1rem 2rem !important;
}
button[data-baseweb="tab"][aria-selected="true"] { 
    color: #818cf8 !important; 
}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] { 
    background: #6366f1 !important; height: 3px !important; 
}

/* ── Buttons ── */
.stButton > button[kind="primary"] {
    background: #4f46e5 !important;
    border: none !important; border-radius: 18px !important;
    padding: 1rem 2.5rem !important;
    font-weight: 800 !important; color: white !important;
    box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    background: #6366f1 !important;
    transform: translateY(-1px);
    box-shadow: 0 15px 20px -5px rgba(99, 102, 241, 0.4) !important;
}

.stDownloadButton button {
    background: #233050 !important;
    border: 1px solid #3e5070 !important;
    border-radius: 18px !important;
    color: #e2e8f0 !important; font-weight: 800 !important;
    padding: 0.8rem 1.5rem !important;
}
.stDownloadButton button:hover {
    border-color: #6366f1 !important;
    color: #ffffff !important;
    background: #2e4060 !important;
}

[data-testid="stExpander"] {
    background: #233050 !important;
    border: 1px solid #3e5070 !important;
    border-radius: 20px !important;
}

/* ── Scrollbars ── */
::-webkit-scrollbar { width: 10px; }
::-webkit-scrollbar-thumb { background: #3e5070; border-radius: 10px; }
::-webkit-scrollbar-track { background: #1a2540; }

</style>
"""

def inject_styles() -> None:
    import streamlit as st
    st.markdown(MAIN_CSS, unsafe_allow_html=True)
