"""
resq_ai.ui.theme
================
Modern, high-contrast, professional design system for ResQ-AI Incident Command Center.
Implements structured layout tokens, high-contrast visible buttons in both light & dark modes,
symmetrical cards, tabular typography, and seamless dark mode elevation.
"""

from __future__ import annotations


def get_css(theme: str = "light") -> str:
    """Returns CSS string with complete styling tokens, dark-mode button illumination, and structured layout."""
    is_dark = (theme == "dark")

    # Backgrounds & Text
    bg_page = "#0B1120" if is_dark else "#F8FAFC"
    bg_sidebar = "#0F172A" if is_dark else "#F1F5F9"
    bg_card = "#1E293B" if is_dark else "#FFFFFF"
    bg_subtle = "#28354D" if is_dark else "#F1F5F9"
    text_primary = "#F8FAFC" if is_dark else "#0F172A"
    text_secondary = "#94A3B8" if is_dark else "#475569"
    border_color = "#334155" if is_dark else "#CBD5E1"

    # Buttons in Dark Mode: Clearly elevated slate gradient with bright top highlight, NEVER flat/black!
    # Buttons in Light Mode: Crisp white with clean slate border
    btn_bg = "linear-gradient(180deg, #2D3E56 0%, #1E293B 100%)" if is_dark else "#FFFFFF"
    btn_border = "#475569" if is_dark else "#CBD5E1"
    btn_text = "#F8FAFC" if is_dark else "#0F172A"
    btn_hover_bg = "linear-gradient(180deg, #3A4F6E 0%, #28374F 100%)" if is_dark else "#F0FDF9"
    btn_hover_border = "#38BDF8" if is_dark else "#0D7A73"
    btn_hover_text = "#38BDF8" if is_dark else "#0D7A73"
    btn_hover_shadow = "0 0 14px rgba(56, 189, 248, 0.35)" if is_dark else "0 2px 6px rgba(13, 122, 115, 0.15)"

    # Segmented Control Tabs
    seg_bg = "#162030" if is_dark else "#F1F5F9"
    seg_border = "#334155" if is_dark else "#CBD5E1"
    seg_text = "#94A3B8" if is_dark else "#475569"
    seg_hover_bg = "#223147" if is_dark else "#E2E8F0"
    seg_active_bg = "linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%)" if is_dark else "linear-gradient(135deg, #0D7A73 0%, #0A655F 100%)"

    # Accents & Priorities
    accent_teal = "#38BDF8" if is_dark else "#0D7A73"
    accent_hover = "#7DD3FC" if is_dark else "#0A655F"
    accent_soft = "#162E47" if is_dark else "#EEF7F6"

    p1_color = "#F87171" if is_dark else "#DC2626"
    p1_soft = "#451A1A" if is_dark else "#FEF2F2"

    p2_color = "#FBBF24" if is_dark else "#D97706"
    p2_soft = "#422C12" if is_dark else "#FFFBEB"

    p3_color = "#60A5FA" if is_dark else "#2563EB"
    p3_soft = "#172E4C" if is_dark else "#EFF6FF"

    amb_color = "#4ADE80" if is_dark else "#16A34A"
    boat_color = "#22D3EE" if is_dark else "#0891B2"

    return f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {{
        --bg-page: {bg_page};
        --bg-sidebar: {bg_sidebar};
        --bg-card: {bg_card};
        --bg-subtle: {bg_subtle};
        --text-primary: {text_primary};
        --text-secondary: {text_secondary};
        --border: {border_color};
        --btn-bg: {btn_bg};
        --btn-border: {btn_border};
        --btn-text: {btn_text};
        --btn-hover-bg: {btn_hover_bg};
        --btn-hover-border: {btn_hover_border};
        --btn-hover-text: {btn_hover_text};
        --btn-hover-shadow: {btn_hover_shadow};
        --seg-bg: {seg_bg};
        --seg-border: {seg_border};
        --seg-text: {seg_text};
        --seg-hover-bg: {seg_hover_bg};
        --seg-active-bg: {seg_active_bg};
        --accent: {accent_teal};
        --accent-hover: {accent_hover};
        --accent-soft: {accent_soft};
        --p1: {p1_color};
        --p1-soft: {p1_soft};
        --p2: {p2_color};
        --p2-soft: {p2_soft};
        --p3: {p3_color};
        --p3-soft: {p3_soft};
        --amb: {amb_color};
        --boat: {boat_color};
    }}

    /* Base Streamlit Overrides */
    .stApp,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stMain"],
    div[data-testid="stMainBlockContainer"] {{
        background-color: var(--bg-page) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
    }}

    /* Global Layout Cleanup */
    header[data-testid="stHeader"],
    div[data-testid="stToolbar"],
    div[data-testid="stDecoration"] {{
        display: none !important;
    }}
    #MainMenu, footer {{
        visibility: hidden !important;
    }}
    .block-container {{
        max-width: 1600px !important;
        padding-top: 0.8rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 1.2rem !important;
        padding-right: 1.2rem !important;
    }}

    /* =======================================================================
       SIDEBAR (SEAMLESS DARK & LIGHT MODE)
       ======================================================================= */
    section[data-testid="stSidebar"],
    div[data-testid="stSidebarContent"],
    div[data-testid="stSidebarUserContent"],
    section[data-testid="stSidebar"] > div {{
        background-color: var(--bg-sidebar) !important;
        color: var(--text-primary) !important;
        border-right: 1px solid var(--border) !important;
    }}

    /* Sidebar Section Titles */
    .sidebar-section-title {{
        color: var(--text-secondary) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        margin-top: 14px !important;
        margin-bottom: 6px !important;
    }}

    /* Sidebar General Text, Captions & Labels */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label {{
        color: var(--text-primary) !important;
    }}
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] .stCaption p {{
        color: var(--text-secondary) !important;
    }}

    /* Sidebar Radio Cards (Scenario Selector) */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 8px 12px !important;
        margin-bottom: 6px !important;
        color: var(--text-primary) !important;
        transition: all 0.15s ease-in-out !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10) !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        border-color: var(--accent) !important;
        transform: translateY(-1px) !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked),
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-checked="true"] {{
        border-color: var(--accent) !important;
        background-color: var(--accent-soft) !important;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
    }}

    /* Sidebar Checkboxes (Map Layers) */
    section[data-testid="stSidebar"] div[data-testid="stCheckbox"] label span {{
        color: var(--text-primary) !important;
        font-size: 0.82rem !important;
    }}

    /* Sidebar Slider */
    section[data-testid="stSidebar"] div[data-testid="stSlider"] div[data-baseweb="slider"] {{
        color: var(--accent) !important;
    }}

    /* Sidebar Collapse Icon */
    button[data-testid="stSidebarCollapseButton"] {{
        color: var(--text-secondary) !important;
    }}
    button[data-testid="stSidebarCollapseButton"]:hover {{
        color: var(--accent) !important;
    }}

    /* Typography & Tabular Digits */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.01em !important;
    }}
    .tabular-nums, code, pre, .mono {{
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums !important;
    }}

    /* =======================================================================
       STRUCTURE: TOP NAVBAR & HEADER
       ======================================================================= */
    .navbar-wrapper {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 8px 14px;
        margin-bottom: 10px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
        transition: all 0.2s ease-in-out;
    }}
    .app-header {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 16px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 10px;
    }}
    .brand-title {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 1.20rem;
        font-weight: 800;
        color: var(--text-primary);
        display: flex;
        align-items: center;
        gap: 8px;
        margin: 0;
    }}
    .brand-subtitle {{
        font-size: 0.74rem;
        color: var(--text-secondary);
        font-weight: 500;
        margin-top: 1px;
    }}

    /* Toggle Switches - No awkward text wrapping */
    div[data-testid="stToggle"] label,
    div[data-testid="stToggle"] span,
    div[data-testid="stToggle"] p {{
        white-space: nowrap !important;
        font-size: 0.80rem !important;
        color: var(--text-primary) !important;
    }}
    div[data-testid="stToggle"] {{
        align-items: center !important;
    }}

    /* Status Pills */
    .hazard-pill {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: var(--bg-subtle);
        border: 1px solid var(--border);
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.80rem;
        font-weight: 600;
        color: var(--text-primary);
    }}
    .status-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
    }}
    .dot-safe {{ background-color: var(--amb); box-shadow: 0 0 6px rgba(74, 222, 128, 0.5); }}
    .dot-advisory {{ background-color: var(--p2); box-shadow: 0 0 6px rgba(251, 191, 36, 0.5); }}
    .dot-critical {{ background-color: var(--p1); box-shadow: 0 0 8px rgba(248, 113, 113, 0.7); }}
    .dot-evacuate {{ background-color: #DC2626; box-shadow: 0 0 10px rgba(220, 38, 38, 0.8); }}

    /* =======================================================================
       STRUCTURE: BOTTOM FOOTER
       ======================================================================= */
    .app-footer-wrapper {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px 18px;
        margin-top: 14px;
        margin-bottom: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }}

    /* =======================================================================
       BUTTON SYSTEM (HIGH-CONTRAST & ELEVATED IN BOTH DARK & LIGHT MODES)
       ======================================================================= */
    div[data-testid="stHorizontalBlock"] {{
        align-items: center !important;
    }}

    /* Secondary / Default Buttons: Clearly elevated, visible border & text in dark mode */
    .stButton > button,
    button[data-testid="stBaseButton-secondary"],
    button[data-testid="baseButton-secondary"],
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {{
        background: var(--btn-bg) !important;
        background-image: var(--btn-bg) !important;
        color: var(--btn-text) !important;
        border: 1px solid var(--btn-border) !important;
        border-radius: 8px !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        line-height: 1.4 !important;
        padding: 6px 12px !important;
        min-height: 36px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.22), inset 0 1px 0 rgba(255, 255, 255, 0.10) !important;
        transition: all 0.16s ease-in-out !important;
    }}

    /* Target inner typography so it inherits button font color */
    .stButton > button p,
    button[data-testid*="stBaseButton"] p,
    button[data-testid*="baseButton"] p,
    div[data-testid="stDownloadButton"] > button p,
    .stButton > button span,
    button[data-testid*="stBaseButton"] span,
    button[data-testid*="baseButton"] span,
    div[data-testid="stDownloadButton"] > button span,
    .stButton > button div,
    button[data-testid*="stBaseButton"] div {{
        color: inherit !important;
        font-weight: inherit !important;
    }}

    /* Secondary Button Hover: Luminous, vibrant highlight */
    .stButton > button:hover,
    button[data-testid="stBaseButton-secondary"]:hover,
    button[data-testid="baseButton-secondary"]:hover,
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        background: var(--btn-hover-bg) !important;
        background-image: var(--btn-hover-bg) !important;
        border-color: var(--btn-hover-border) !important;
        color: var(--btn-hover-text) !important;
        transform: translateY(-1px) !important;
        box-shadow: var(--btn-hover-shadow) !important;
    }}

    .stButton > button:active,
    button[data-testid="stBaseButton-secondary"]:active,
    div[data-testid="stDownloadButton"] > button:active {{
        transform: translateY(0px) !important;
        filter: brightness(0.95);
    }}

    /* Primary Action & Submit Buttons: Vibrant gradient, high contrast */
    .stButton > button[kind="primary"],
    button[data-testid="stBaseButton-primary"],
    button[data-testid="baseButton-primary"],
    button[data-testid="stBaseButton-secondaryFormSubmit"],
    .stFormSubmitButton > button {{
        background: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        background-image: linear-gradient(135deg, #0EA5E9 0%, #0284C7 100%) !important;
        color: #FFFFFF !important;
        border: 1px solid #38BDF8 !important;
        border-radius: 8px !important;
        font-weight: 700 !important;
        font-size: 0.84rem !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.35) !important;
    }}

    /* Primary Button Hover */
    .stButton > button[kind="primary"]:hover,
    button[data-testid="stBaseButton-primary"]:hover,
    button[data-testid="baseButton-primary"]:hover,
    button[data-testid="stBaseButton-secondaryFormSubmit"]:hover,
    .stFormSubmitButton > button:hover {{
        background: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%) !important;
        background-image: linear-gradient(135deg, #38BDF8 0%, #0284C7 100%) !important;
        border-color: #7DD3FC !important;
        color: #FFFFFF !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 14px rgba(56, 189, 248, 0.5) !important;
    }}

    /* Download Buttons */
    div[data-testid="stDownloadButton"] > button {{
        width: 100% !important;
    }}

    /* =======================================================================
       CARD CONTAINERS & KPIS
       ======================================================================= */
    .kpi-box {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 16px;
        min-height: 105px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }}
    .kpi-title {{
        font-size: 0.72rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}
    .kpi-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.50rem;
        font-weight: 800;
        color: var(--text-primary);
        font-variant-numeric: tabular-nums;
        margin: 2px 0;
    }}
    .kpi-footer {{
        font-size: 0.74rem;
        color: var(--text-secondary);
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Algorithm Cards */
    .algo-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 12px 14px;
        min-height: 125px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .algo-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 4px;
    }}
    .algo-name {{
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-primary);
    }}
    .algo-tag {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.66rem;
        font-weight: 700;
        padding: 2px 6px;
        border-radius: 4px;
        background: var(--accent-soft);
        color: var(--accent);
        border: 1px solid var(--border);
    }}
    .algo-desc {{
        font-size: 0.76rem;
        color: var(--text-secondary);
        line-height: 1.35;
        margin-bottom: 6px;
    }}
    .algo-footer {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.70rem;
        color: var(--text-secondary);
        border-top: 1px solid var(--border);
        padding-top: 5px;
        display: flex;
        justify-content: space-between;
    }}

    /* Priority Chips */
    .chip-p1 {{
        background: var(--p1-soft);
        color: var(--p1);
        border: 1px solid var(--p1);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .chip-p2 {{
        background: var(--p2-soft);
        color: var(--p2);
        border: 1px solid var(--p2);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}
    .chip-p3 {{
        background: var(--p3-soft);
        color: var(--p3);
        border: 1px solid var(--p3);
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}

    /* =======================================================================
       FORM INPUTS & SEGMENTED TABS (HIGH CONTRAST)
       ======================================================================= */
    div[data-testid="stSegmentedControl"] {{
        background-color: var(--bg-subtle) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        padding: 3px !important;
        margin-bottom: 8px !important;
        width: 100% !important;
        display: flex !important;
        gap: 3px !important;
    }}
    /* Segmented Control Inactive Tabs */
    div[data-testid="stSegmentedControl"] button,
    button[data-testid="stBaseButton-segmented_control"] {{
        flex: 1 1 auto !important;
        border: 1px solid var(--seg-border) !important;
        border-radius: 6px !important;
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.78rem !important;
        color: var(--seg-text) !important;
        background: var(--seg-bg) !important;
        background-color: var(--seg-bg) !important;
        background-image: none !important;
        padding: 5px 8px !important;
        min-height: 32px !important;
        box-shadow: none !important;
        transition: all 0.15s ease-in-out !important;
    }}
    div[data-testid="stSegmentedControl"] button:hover,
    button[data-testid="stBaseButton-segmented_control"]:hover {{
        color: var(--text-primary) !important;
        background: var(--seg-hover-bg) !important;
        background-color: var(--seg-hover-bg) !important;
        border-color: var(--accent) !important;
    }}
    div[data-testid="stSegmentedControl"] button p,
    button[data-testid="stBaseButton-segmented_control"] p {{
        color: inherit !important;
    }}

    /* Segmented Control Active Tab */
    div[data-testid="stSegmentedControl"] button[aria-checked="true"],
    div[data-testid="stSegmentedControl"] button[data-checked="true"],
    button[data-testid="stBaseButton-segmented_controlActive"] {{
        background: var(--seg-active-bg) !important;
        background-image: var(--seg-active-bg) !important;
        color: #FFFFFF !important;
        border: 1px solid var(--accent) !important;
        font-weight: 700 !important;
        box-shadow: 0 2px 8px rgba(14, 165, 233, 0.4) !important;
    }}
    div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
    div[data-testid="stSegmentedControl"] button[data-checked="true"] p,
    button[data-testid="stBaseButton-segmented_controlActive"] p {{
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }}

    /* Text inputs & textareas */
    input[type="text"], input[type="search"], textarea {{
        background-color: var(--bg-card) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 7px !important;
    }}
    input[type="text"]:focus, textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 8px rgba(56, 189, 248, 0.25) !important;
    }}

    /* Container wrapper */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 10px 12px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar {{
        width: 6px;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]::-webkit-scrollbar-thumb {{
        background: var(--border);
        border-radius: 3px;
    }}

    /* Progress bar */
    .progress-bar-bg {{
        background: var(--border);
        height: 4px;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 5px;
    }}
    .progress-bar-fill {{
        height: 100%;
        border-radius: 2px;
    }}

    .stacked-bar {{
        display: flex;
        height: 4px;
        border-radius: 2px;
        overflow: hidden;
        margin-top: 5px;
        margin-bottom: 5px;
    }}
</style>
"""


# Legacy theme constant for compatibility
TACTICAL_CSS = get_css("light")
