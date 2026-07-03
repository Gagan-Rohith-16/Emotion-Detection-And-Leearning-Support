"""Visual theme primitives for the premium Streamlit interface."""

from __future__ import annotations

import streamlit as st


def apply_theme(dark_mode: bool = False) -> None:
    """Inject the responsive color system, cards, animations, and controls."""

    palette = (
        {
            "background": "#090D18",
            "surface": "rgba(18, 25, 43, 0.86)",
            "surface_strong": "#151D31",
            "text": "#F5F7FF",
            "muted": "#A7B0C8",
            "border": "rgba(255, 255, 255, 0.10)",
            "shadow": "rgba(0, 0, 0, 0.30)",
        }
        if dark_mode
        else {
            "background": "#F5F7FC",
            "surface": "rgba(255, 255, 255, 0.88)",
            "surface_strong": "#FFFFFF",
            "text": "#151A2D",
            "muted": "#626B82",
            "border": "rgba(30, 41, 79, 0.10)",
            "shadow": "rgba(49, 46, 129, 0.10)",
        }
    )
    # CSS is centralized here so pages share a consistent visual language.
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@600;700;800&display=swap');

        :root {{
            --app-bg: {palette['background']};
            --surface: {palette['surface']};
            --surface-strong: {palette['surface_strong']};
            --text: {palette['text']};
            --muted: {palette['muted']};
            --border: {palette['border']};
            --shadow: {palette['shadow']};
            --primary: #625BFF;
            --primary-2: #8B5CF6;
            --cyan: #19C6D1;
            --success: #19B982;
        }}

        html,
        body,
        [data-testid="stAppViewContainer"],
        [data-testid="stSidebar"],
        .stMarkdown,
        label,
        p,
        span,
        div,
        h1,
        h2,
        h3,
        h4,
        h5,
        h6 {{
            font-family: 'DM Sans', sans-serif;
            color: var(--text) !important;
        }}
        [data-testid="stAppViewContainer"] {{
            background:
                radial-gradient(circle at 8% 4%, rgba(98,91,255,.15), transparent 26rem),
                radial-gradient(circle at 92% 12%, rgba(25,198,209,.12), transparent 24rem),
                var(--app-bg);
        }}
        #MainMenu {{ visibility: hidden; }}
        [data-testid="stToolbar"] {{ display: none !important; }}
        [data-testid="stDecoration"] {{ display: none !important; }}
        [data-testid="stAppViewContainer"] {{ color-scheme: light; }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: var(--surface-strong);
            border-right: 1px solid var(--border);
        }}
        .block-container {{ max-width: 1220px; padding-top: 2rem; padding-bottom: 4rem; }}
        h1, h2, h3 {{ font-family: 'Manrope', sans-serif; letter-spacing: -.035em; }}

        .brand-mark {{
            display: inline-flex; align-items: center; justify-content: center;
            width: 2.45rem; height: 2.45rem; border-radius: .85rem;
            color: white; font-size: 1.2rem;
            background: linear-gradient(135deg, var(--primary), var(--cyan));
            box-shadow: 0 10px 28px rgba(98,91,255,.28);
        }}
        .eyebrow {{ color: var(--primary); font-weight: 700; font-size: .78rem;
            letter-spacing: .13em; text-transform: uppercase; margin-bottom: .55rem; }}
        .hero-title {{ font-family: 'Manrope', sans-serif; font-size: clamp(2.35rem, 5vw, 4.6rem);
            line-height: 1.02; font-weight: 800; letter-spacing: -.065em; margin: .2rem 0 1rem; }}
        .gradient-text {{ background: linear-gradient(100deg, #625BFF, #A855F7 52%, #19C6D1);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
        .hero-copy {{ max-width: 720px; color: var(--muted); font-size: 1.08rem; line-height: 1.75; }}

        .glass-card {{
            height: 100%; padding: 1.35rem; border-radius: 1.25rem;
            background: var(--surface); border: 1px solid var(--border);
            box-shadow: 0 18px 50px var(--shadow); backdrop-filter: blur(16px);
            animation: rise .45s ease both; transition: transform .2s ease, border-color .2s ease;
        }}
        .glass-card:hover {{ transform: translateY(-3px); border-color: rgba(98,91,255,.35); }}
        .card-icon {{ font-size: 1.35rem; margin-bottom: .75rem; }}
        .card-title {{ font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 1.05rem; margin-bottom: .4rem; }}
        .card-copy {{ color: var(--muted); font-size: .9rem; line-height: 1.6; }}
        .metric-value {{ font: 800 2rem 'Manrope', sans-serif; letter-spacing: -.05em; }}
        .metric-label {{ color: var(--muted); font-size: .82rem; margin-top: .2rem; }}

        .status-pill {{ display: inline-flex; align-items: center; gap: .45rem; padding: .42rem .75rem;
            border-radius: 999px; color: var(--success); background: rgba(25,185,130,.1);
            border: 1px solid rgba(25,185,130,.2); font-size: .78rem; font-weight: 700; }}
        .dot {{ width: .48rem; height: .48rem; border-radius: 50%; background: currentColor;
            box-shadow: 0 0 0 .25rem rgba(25,185,130,.12); }}
        .section-heading {{ font: 700 1.7rem 'Manrope', sans-serif; margin: 2.6rem 0 .35rem; }}
        .section-copy {{ color: var(--muted); margin-bottom: 1.25rem; }}

        div.stButton > button, div.stFormSubmitButton > button {{
            border: 0; border-radius: .8rem; font-weight: 700; min-height: 2.8rem;
            color: white; background: linear-gradient(100deg, var(--primary), var(--primary-2));
            box-shadow: 0 10px 24px rgba(98,91,255,.2); transition: all .2s ease;
        }}
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {{
            color: white; transform: translateY(-1px); box-shadow: 0 14px 28px rgba(98,91,255,.28);
        }}
        [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {{
            border-radius: .8rem !important; border-color: var(--border) !important;
        }}
        div[data-testid="stProgress"] > div > div > div {{
            background: linear-gradient(90deg, var(--primary), var(--cyan));
        }}
        footer {{ visibility: hidden; }}
        @keyframes rise {{ from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }} }}
        @media (max-width: 700px) {{
            .block-container {{ padding: 1.2rem 1rem 3rem; }}
            .hero-title {{ font-size: 2.5rem; }}
            .glass-card {{ padding: 1.05rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

