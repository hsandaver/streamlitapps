"""
app.py

A refactored, design-forward Streamlit LAB Color Identifier.
Features modular data handling, refined visualizations, and a premium UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union, List, Tuple, Optional, Any, IO, Dict
import logging
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor, sRGBColor
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF

# =============================================================================
# Custom Exceptions
# =============================================================================

class InputError(Exception):
    """Exception raised for errors in the input LAB color."""


class DatasetError(Exception):
    """Exception raised for errors in the dataset file."""


class RDFParsingError(Exception):
    """Exception raised for errors during RDF file parsing."""


class ConversionError(Exception):
    """Exception raised for errors in color conversions."""

# =============================================================================
# Setup and Configuration
# =============================================================================

logging.basicConfig(
    filename="color_analyzer.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    level=logging.DEBUG,
)
logger = logging.getLogger(__name__)

# Compatibility for older NumPy versions
if not hasattr(np, "asscalar"):
    np.asscalar = lambda a: a.item() if isinstance(a, np.ndarray) else a

REQUIRED_COLUMNS = {"L", "A", "B", "Color Name"}

THEME = {
    "bg_top": "#fdf6ef",
    "bg_bottom": "#eef3f6",
    "ink": "#1b1b1f",
    "muted": "#5c5f6a",
    "accent": "#f05d5e",
    "accent_soft": "#f7b4a8",
    "accent_2": "#0f6b6c",
    "accent_3": "#f2c14e",
    "card": "rgba(255, 255, 255, 0.78)",
    "card_border": "rgba(24, 24, 28, 0.12)",
    "grid": "rgba(24, 24, 28, 0.08)",
    "shadow": "0 18px 45px rgba(18, 20, 26, 0.14)",
    "glow": "0 18px 45px rgba(240, 93, 94, 0.22)",
}

FONT_BODY = "Space Grotesk"
FONT_DISPLAY = "Fraunces"

# =============================================================================
# Data Models
# =============================================================================

@dataclass
class ColorMatchResult:
    input_lab: List[float]
    closest_lab: List[float]
    closest_name: str
    delta_e: float
    input_rgb: Tuple[int, int, int]
    closest_rgb: Tuple[int, int, int]
    method: str

    @property
    def delta_e_label(self) -> str:
        return f"{self.delta_e:.2f}"

    @property
    def method_label(self) -> str:
        return "CIEDE2000" if self.method == "ciede2000" else "Euclidean Delta E 76"

# =============================================================================
# Styling
# =============================================================================

def inject_global_styles() -> None:
    """Injects CSS for a premium, gallery-like aesthetic."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@300;500;700&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

        :root {{
            --bg-top: {THEME['bg_top']};
            --bg-bottom: {THEME['bg_bottom']};
            --ink: {THEME['ink']};
            --muted: {THEME['muted']};
            --accent: {THEME['accent']};
            --accent-soft: {THEME['accent_soft']};
            --accent-2: {THEME['accent_2']};
            --accent-3: {THEME['accent_3']};
            --card: {THEME['card']};
            --card-border: {THEME['card_border']};
            --grid: {THEME['grid']};
            --shadow: {THEME['shadow']};
            --glow: {THEME['glow']};
        }}

        html, body, [class*="st-"] {{
            font-family: '{FONT_BODY}', sans-serif;
            color: var(--ink);
        }}

        .stApp {{
            background: radial-gradient(1200px 700px at 12% -10%, rgba(240, 93, 94, 0.16), transparent 60%),
                        radial-gradient(1000px 600px at 90% 10%, rgba(15, 107, 108, 0.18), transparent 65%),
                        linear-gradient(160deg, var(--bg-top), var(--bg-bottom));
        }}

        .stApp::before {{
            content: "";
            position: fixed;
            inset: -40px;
            background: radial-gradient(300px 300px at 75% 15%, rgba(242, 193, 78, 0.18), transparent 65%),
                        radial-gradient(280px 280px at 20% 90%, rgba(240, 93, 94, 0.12), transparent 70%);
            pointer-events: none;
            z-index: 0;
        }}

        .block-container {{
            max-width: 1200px;
            padding-top: 2.5rem;
            padding-bottom: 4rem;
        }}

        h1, h2, h3 {{
            font-family: '{FONT_DISPLAY}', serif;
            letter-spacing: -0.02em;
        }}

        h1 {{
            font-size: 3rem;
            margin-bottom: 0.25rem;
        }}

        .hero {{
            display: grid;
            grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr);
            gap: 2rem;
            align-items: center;
            padding: 2.5rem;
            border-radius: 28px;
            background: var(--card);
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
            position: relative;
            overflow: hidden;
            animation: rise 700ms ease-out;
        }}

        .hero::after {{
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(240, 93, 94, 0.12), transparent 55%);
            opacity: 0.6;
            pointer-events: none;
        }}

        .hero-kicker {{
            text-transform: uppercase;
            letter-spacing: 0.26em;
            font-size: 0.7rem;
            color: var(--muted);
            margin-bottom: 0.75rem;
        }}

        .hero-body p {{
            font-size: 1.05rem;
            color: var(--muted);
            line-height: 1.7;
        }}

        .hero-tags {{
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin-top: 1.4rem;
        }}

        .chip {{
            padding: 0.35rem 0.85rem;
            border-radius: 999px;
            border: 1px solid var(--card-border);
            background: rgba(255, 255, 255, 0.65);
            font-size: 0.75rem;
            letter-spacing: 0.02em;
        }}

        .hero-card {{
            background: rgba(17, 20, 28, 0.88);
            color: #f9f9fb;
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: var(--glow);
        }}

        .hero-label {{
            font-size: 0.8rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: rgba(249, 249, 251, 0.6);
        }}

        .hero-swatch {{
            width: 100%;
            height: 110px;
            border-radius: 18px;
            margin: 1rem 0;
            border: 1px solid rgba(255, 255, 255, 0.25);
        }}

        .hero-meta {{
            display: grid;
            gap: 0.4rem;
            font-size: 0.85rem;
            color: rgba(249, 249, 251, 0.78);
        }}

        .section-heading {{
            margin-top: 2.5rem;
            margin-bottom: 1.4rem;
        }}

        .section-title {{
            font-family: '{FONT_DISPLAY}', serif;
            font-size: 1.8rem;
            margin-bottom: 0.35rem;
        }}

        .section-subtitle {{
            color: var(--muted);
        }}

        .glass-card {{
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 22px;
            padding: 1.5rem;
            box-shadow: var(--shadow);
        }}

        .color-card {{
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 1rem;
            align-items: center;
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.1rem;
            box-shadow: var(--shadow);
        }}

        .color-swatch {{
            width: 100%;
            height: 110px;
            border-radius: 18px;
            border: 1px solid rgba(0, 0, 0, 0.12);
        }}

        .color-card-title {{
            font-size: 1.1rem;
            font-weight: 600;
            margin-bottom: 0.2rem;
        }}

        .color-card-subtitle {{
            font-size: 0.85rem;
            color: var(--muted);
        }}

        .color-card-hex {{
            font-size: 1.1rem;
            letter-spacing: 0.1em;
            margin: 0.4rem 0;
        }}

        .color-card-lab {{
            font-size: 0.85rem;
            color: var(--muted);
        }}

        .blend-card {{
            background: var(--card);
            border: 1px solid var(--card-border);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: var(--shadow);
        }}

        .blend-bar {{
            height: 18px;
            border-radius: 999px;
            margin-bottom: 0.6rem;
        }}

        .blend-meta {{
            font-size: 0.8rem;
            color: var(--muted);
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }}

        .status-pill {{
            display: inline-flex;
            gap: 0.4rem;
            align-items: center;
            padding: 0.3rem 0.8rem;
            border-radius: 999px;
            background: rgba(15, 107, 108, 0.12);
            border: 1px solid rgba(15, 107, 108, 0.2);
            color: var(--accent-2);
            font-size: 0.78rem;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }}

        .empty-state {{
            margin-top: 2rem;
            padding: 2rem;
            border-radius: 24px;
            border: 1px dashed var(--card-border);
            background: rgba(255, 255, 255, 0.65);
            text-align: center;
            color: var(--muted);
        }}

        div[data-testid="stMetric"] {{
            background: var(--card);
            padding: 1.1rem 1rem;
            border-radius: 18px;
            border: 1px solid var(--card-border);
            box-shadow: var(--shadow);
        }}

        div[data-testid="stMetric"] label {{
            color: var(--muted);
        }}

        .stButton > button, .stDownloadButton > button {{
            background: linear-gradient(135deg, var(--accent), var(--accent-3));
            color: #fff;
            border: none;
            padding: 0.65rem 1.2rem;
            border-radius: 999px;
            font-weight: 600;
            letter-spacing: 0.02em;
            box-shadow: var(--glow);
            transition: transform 120ms ease, box-shadow 120ms ease;
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            box-shadow: 0 16px 36px rgba(240, 93, 94, 0.28);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.5rem;
            background: rgba(255, 255, 255, 0.7);
            padding: 0.4rem;
            border-radius: 999px;
            border: 1px solid var(--card-border);
        }}

        .stTabs [data-baseweb="tab"] {{
            background: transparent;
            border-radius: 999px;
            padding: 0.35rem 1rem;
            color: var(--muted);
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background: linear-gradient(135deg, var(--accent), var(--accent-3));
            color: #fff;
        }}

        div[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, rgba(17, 20, 28, 0.98), rgba(17, 20, 28, 0.92));
            color: #f2f2f4;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }}

        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {{
            color: #f9f9fb;
        }}

        div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span {{
            color: rgba(249, 249, 251, 0.8);
        }}

        @keyframes rise {{
            from {{
                opacity: 0;
                transform: translateY(18px);
            }}
            to {{
                opacity: 1;
                transform: translateY(0);
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# Utility Functions (Validation, Conversion, Logging)
# =============================================================================

def _log_and_report(message: str, level: str = "debug", error_type: Optional[str] = None) -> None:
    """Logs and reports messages to both the log file and Streamlit UI."""
    if level == "debug":
        logger.debug(message)
    elif level == "info":
        logger.info(message)
        st.info(message)
    elif level == "warning":
        logger.warning(message)
        st.warning(message)
    elif level == "error":
        logger.error(message)
        st.error(f"{error_type}: {message}" if error_type else message)


def validate_lab_color(
    lab: Union[List[float], Tuple[float, float, float], np.ndarray],
    report: bool = True,
) -> bool:
    """Validate that LAB input is a 3-element numeric list within valid ranges."""
    if not isinstance(lab, (list, tuple, np.ndarray)) or len(lab) != 3:
        if report:
            _log_and_report(
                "Input LAB color must be a list, tuple, or array of three numerical values.",
                "error",
                "Input Error",
            )
        return False
    try:
        L, A, B = map(float, lab)
    except (ValueError, TypeError) as exc:
        if report:
            _log_and_report(f"LAB components must be numerical values. {exc}", "error", "Input Error")
        return False
    if not (0 <= L <= 100 and -128 <= A <= 127 and -128 <= B <= 127):
        if report:
            _log_and_report(
                "LAB components are out of range. L: 0-100, A and B: -128 to 127.",
                "error",
                "Input Error",
            )
        return False
    return True


@st.cache_data(show_spinner=False)
def lab_to_rgb(lab_color: Union[List[float], Tuple[float, float, float], np.ndarray]) -> Tuple[int, int, int]:
    """
    Converts a LAB color to RGB using colormath.
    Returns an RGB tuple with values between 0 and 255.
    """
    try:
        lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
        rgb = convert_color(lab, sRGBColor, target_illuminant="d65")
        return tuple(int(max(0, min(1, c)) * 255) for c in [rgb.rgb_r, rgb.rgb_g, rgb.rgb_b])
    except Exception as exc:
        _log_and_report(f"Error converting LAB to RGB: {exc}", "error", "Conversion Error")
        return (0, 0, 0)


def format_rgb(rgb: Tuple[int, int, int]) -> str:
    """Formats an RGB tuple into a Plotly-friendly RGB string."""
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converts an RGB tuple into a hex string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def calculate_delta_e(
    input_lab: Union[List[float], Tuple[float, float, float]],
    dataset_df: pd.DataFrame,
    method: str = "euclidean",
) -> np.ndarray:
    """
    Calculates Delta-E values between input LAB and dataset colors.
    Supports Euclidean (Delta E 76) and CIEDE2000.
    """
    input_lab_arr = np.array(input_lab)
    if method.lower() == "ciede2000":
        input_lab_obj = LabColor(*input_lab_arr)
        delta_e_values = np.array(
            [
                delta_e_cie2000(input_lab_obj, LabColor(*row))
                for row in dataset_df[["L", "A", "B"]].values
            ]
        )
    else:
        delta_e_values = np.linalg.norm(dataset_df[["L", "A", "B"]].values - input_lab_arr, axis=1)
    return delta_e_values


def find_closest_color(
    input_lab: Union[List[float], Tuple[float, float, float]],
    dataset_df: pd.DataFrame,
    delta_e_method: str = "euclidean",
) -> Tuple[Optional[pd.Series], Optional[float]]:
    """
    Finds the closest color from the dataset to the input LAB color based on Delta-E.
    Returns the closest color row and its Delta-E value.
    """
    if dataset_df.empty:
        _log_and_report("Dataset is empty after validation.", "error", "Dataset Error")
        return None, None
    delta_e_values = calculate_delta_e(input_lab, dataset_df, method=delta_e_method)
    if np.all(np.isnan(delta_e_values)):
        _log_and_report(
            "Delta-E calculation resulted in all NaN values. Check dataset and input.",
            "error",
        )
        return None, None
    min_idx = np.nanargmin(delta_e_values)
    min_delta_e = delta_e_values[min_idx]
    closest_color = dataset_df.iloc[min_idx]
    return closest_color, min_delta_e


@st.cache_data(show_spinner=True)
def load_dataset(uploaded_file: IO[Any]) -> pd.DataFrame:
    """
    Loads and validates the CSV dataset containing LAB values and color names.
    Ensures all required columns are present and non-null.
    """
    try:
        dataset_df = pd.read_csv(uploaded_file)
    except Exception as exc:
        _log_and_report(f"Error reading CSV file: {exc}", "error", "File Reading Error")
        raise DatasetError("Failed to load dataset.")
    if not REQUIRED_COLUMNS.issubset(dataset_df.columns):
        missing_cols = REQUIRED_COLUMNS - set(dataset_df.columns)
        _log_and_report(f"CSV is missing required columns: {missing_cols}", "error", "Dataset Error")
        raise DatasetError("Dataset missing required columns.")
    dataset_df = dataset_df.copy()
    dataset_df["Color Name"] = dataset_df["Color Name"].astype(str).str.strip()
    for col in ["L", "A", "B"]:
        dataset_df[col] = pd.to_numeric(dataset_df[col], errors="coerce")
    if dataset_df[list(REQUIRED_COLUMNS)].isnull().any().any():
        _log_and_report("CSV contains missing or non-numeric values in required columns.", "error", "Dataset Error")
        raise DatasetError("Dataset contains invalid values.")
    if dataset_df.empty:
        _log_and_report("CSV contains no valid rows after cleaning.", "error", "Dataset Error")
        raise DatasetError("Dataset is empty.")
    return dataset_df


@st.cache_data(show_spinner=True)
def extract_alternative_terms_rdf(rdf_file: IO[Any]) -> pd.DataFrame:
    """
    Extracts alternative color terms from an RDF/XML file.
    Filters rdfs:label, skos:altLabel, and dc:description while applying digit filters.
    Processes all subjects of type skos:Concept.
    """
    g = Graph()
    try:
        g.parse(file=rdf_file, format="xml")
    except Exception as exc:
        _log_and_report(f"Failed to parse RDF file: {exc}", "error", "RDF Parsing Error")
        raise RDFParsingError("Failed to parse RDF file.")

    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    DC_ELEM = URIRef("http://purl.org/dc/elements/1.1/description")

    subjects = list(
        g.subjects(predicate=RDF.type, object=URIRef("http://www.w3.org/2004/02/skos/core#Concept"))
    )
    if not subjects:
        _log_and_report("No subjects with skos:Concept type found in RDF file.", "warning")
        return pd.DataFrame()

    rows = []

    def add_term(term: Literal, skip_digit_filter: bool = False) -> None:
        if isinstance(term, Literal):
            term_text = str(term).strip()
            if not skip_digit_filter:
                if "centroid" in term_text.lower():
                    rows.append({"Term": term_text, "Language": term.language or "unknown"})
                    return
                if re.search(r"\d", term_text) or len(term_text) > 50:
                    return
            else:
                if len(term_text) > 100:
                    return
            rows.append({"Term": term_text, "Language": term.language or "unknown"})

    for subj in subjects:
        for label in g.objects(subject=subj, predicate=RDFS.label):
            add_term(label, skip_digit_filter=False)
        for alt in g.objects(subject=subj, predicate=SKOS.altLabel):
            add_term(alt, skip_digit_filter=False)

    for desc in g.objects(predicate=DC_ELEM):
        add_term(desc, skip_digit_filter=True)

    return pd.DataFrame(rows).drop_duplicates()

# =============================================================================
# Visualization Functions
# =============================================================================

def apply_plot_theme(fig: go.Figure, for_3d: bool = False) -> go.Figure:
    """Apply a cohesive theme to Plotly figures."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family=FONT_BODY, color=THEME["ink"]),
        title_font=dict(family=FONT_DISPLAY, color=THEME["ink"], size=22),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.7)", bordercolor=THEME["card_border"]),
    )
    if not for_3d:
        fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
        fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], zerolinecolor=THEME["grid"])
    return fig


def truncate_label(label: str, max_length: int = 20) -> str:
    """Truncates a label to a maximum length."""
    return label if len(label) <= max_length else label[:max_length] + "..."


def create_alternative_terms_sunburst(
    df: pd.DataFrame,
    base_color: Optional[str] = None,
    trunc_length: int = 20,
) -> go.Figure:
    """Generates a sunburst chart from alternative terms."""
    if df.empty:
        return go.Figure()

    english_terms = df[df["Language"] == "en"]
    main_term = english_terms["Term"].iloc[0] if not english_terms.empty else df["Term"].iloc[0]

    df_alt = df[df["Term"] != main_term].copy()
    data = [{"id": "root", "parent": "", "name": main_term}]

    for lang in df_alt["Language"].unique():
        data.append({"id": f"lang_{lang}", "parent": "root", "name": lang})

    for idx, row in df_alt.iterrows():
        data.append({"id": str(idx), "parent": f"lang_{row['Language']}", "name": row["Term"]})

    sunburst_df = pd.DataFrame(data)
    sunburst_df["full_name"] = sunburst_df["name"]
    sunburst_df["name"] = sunburst_df["name"].apply(lambda x: truncate_label(x, trunc_length))

    fig = px.sunburst(
        sunburst_df,
        ids="id",
        names="name",
        parents="parent",
        custom_data=["full_name"],
        title="Alternative Terms Sunburst",
        template="plotly_white",
        color="id" if not base_color else None,
        color_discrete_sequence=[base_color] if base_color else [THEME["accent"], THEME["accent_2"], THEME["accent_3"]],
    )
    fig.update_traces(
        textfont=dict(size=16),
        insidetextorientation="radial",
        hovertemplate="<b>%{customdata[0]}</b><extra></extra>",
    )
    fig = apply_plot_theme(fig)
    fig.update_layout(margin=dict(l=30, r=30, t=70, b=30))
    return fig


def create_color_comparison_plot(
    input_rgb: Tuple[int, int, int],
    closest_rgb: Tuple[int, int, int],
    input_lab: List[float],
    closest_lab: List[float],
    closest_color_name: str,
    delta_e: float,
) -> go.Figure:
    """Creates a side-by-side scatter plot comparing input and closest colors."""
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[0],
                y=[1],
                mode="markers",
                marker=dict(size=70, color=format_rgb(input_rgb), line=dict(width=2, color=THEME["ink"])),
                name="Input Color",
                hovertemplate=(
                    f"Input LAB: L={input_lab[0]:.2f}, A={input_lab[1]:.2f}, B={input_lab[2]:.2f}"
                    f"<br>Delta-E: {delta_e:.2f}<extra></extra>"
                ),
            ),
            go.Scatter(
                x=[1],
                y=[1],
                mode="markers",
                marker=dict(size=70, color=format_rgb(closest_rgb), line=dict(width=2, color=THEME["ink"])),
                name=f"Closest: {closest_color_name}",
                hovertemplate=(
                    f"Closest LAB: L={closest_lab[0]:.2f}, A={closest_lab[1]:.2f}, B={closest_lab[2]:.2f}"
                    f"<extra></extra>"
                ),
            ),
        ]
    )
    fig.update_layout(
        title="Input vs Closest Match",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 1.5]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 1.5]),
        showlegend=False,
        annotations=[
            dict(
                x=0,
                y=1.55,
                text="Input Color",
                showarrow=False,
                font=dict(size=14, color=THEME["ink"]),
                xanchor="center",
            ),
            dict(
                x=1,
                y=1.55,
                text=f"Closest: {closest_color_name}",
                showarrow=False,
                font=dict(size=14, color=THEME["ink"]),
                xanchor="center",
            ),
        ],
    )
    return apply_plot_theme(fig)


def create_lab_comparison_bar(
    input_lab: List[float],
    closest_lab: List[float],
    closest_color_name: str,
    input_rgb: Tuple[int, int, int],
    closest_rgb: Tuple[int, int, int],
) -> go.Figure:
    """Creates a bar chart comparing LAB components."""
    components = ["L", "A", "B"]
    data = pd.DataFrame(
        {
            "Component": components * 2,
            "Value": input_lab + closest_lab,
            "Type": ["Input LAB"] * 3 + [f"Closest LAB: {closest_color_name}"] * 3,
        }
    )
    color_map = {
        "Input LAB": format_rgb(input_rgb),
        f"Closest LAB: {closest_color_name}": format_rgb(closest_rgb),
    }
    fig = px.bar(
        data_frame=data,
        x="Component",
        y="Value",
        color="Type",
        barmode="group",
        hover_data=["Value"],
        title="LAB Component Comparison",
        template="plotly_white",
        color_discrete_map=color_map,
    )
    for i, component in enumerate(components):
        delta = abs(input_lab[i] - closest_lab[i])
        fig.add_annotation(
            x=component,
            y=max(input_lab[i], closest_lab[i]) + 5,
            text=f"Delta: {delta:.2f}",
            showarrow=False,
            font=dict(size=12, color=THEME["muted"]),
        )
    fig.update_layout(xaxis_title="LAB Components", yaxis_title="Values", legend_title="Color Type")
    return apply_plot_theme(fig)


def create_3d_lab_plot(
    input_lab: List[float],
    closest_lab: List[float],
    closest_color_name: str,
    dataset_df: pd.DataFrame,
    input_rgb: Tuple[int, int, int],
    closest_rgb: Tuple[int, int, int],
) -> go.Figure:
    """Generates a 3D scatter plot of the LAB color space."""
    dataset_points = go.Scatter3d(
        x=dataset_df["L"],
        y=dataset_df["A"],
        z=dataset_df["B"],
        mode="markers",
        marker=dict(size=3, color="rgba(120,120,130,0.35)", opacity=0.5),
        name="Dataset Colors",
        hoverinfo="text",
        text=dataset_df["Color Name"],
    )
    input_point = go.Scatter3d(
        x=[input_lab[0]],
        y=[input_lab[1]],
        z=[input_lab[2]],
        mode="markers+text",
        marker=dict(size=10, color=format_rgb(input_rgb), opacity=1),
        text=["Input Color"],
        textposition="top center",
        name="Input Color",
        hoverinfo="text",
    )
    closest_point = go.Scatter3d(
        x=[closest_lab[0]],
        y=[closest_lab[1]],
        z=[closest_lab[2]],
        mode="markers+text",
        marker=dict(size=10, color=format_rgb(closest_rgb), opacity=1),
        text=[f"Closest: {closest_color_name}"],
        textposition="top center",
        name="Closest Color",
        hoverinfo="text",
    )
    fig = go.Figure(data=[dataset_points, input_point, closest_point])
    fig.update_layout(
        title="3D LAB Color Space",
        scene=dict(
            xaxis_title="L",
            yaxis_title="A",
            zaxis_title="B",
            xaxis=dict(range=[0, 100], backgroundcolor="rgba(255,255,255,0.7)", gridcolor=THEME["grid"]),
            yaxis=dict(range=[-128, 127], backgroundcolor="rgba(255,255,255,0.7)", gridcolor=THEME["grid"]),
            zaxis=dict(range=[-128, 127], backgroundcolor="rgba(255,255,255,0.7)", gridcolor=THEME["grid"]),
            bgcolor="rgba(0,0,0,0)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        ),
        legend=dict(x=0.7, y=0.9),
        margin=dict(l=0, r=0, t=70, b=0),
    )
    return apply_plot_theme(fig, for_3d=True)


def create_delta_e_histogram(delta_e_values: np.ndarray) -> go.Figure:
    """Creates a histogram of Delta-E values across the dataset."""
    fig = px.histogram(
        x=delta_e_values,
        nbins=30,
        title="Delta-E Distribution",
        labels={"x": "Delta-E Value", "y": "Count"},
        template="plotly_white",
        opacity=0.8,
        color_discrete_sequence=[THEME["accent"]],
    )
    fig.update_layout(xaxis_title="Delta-E", yaxis_title="Frequency")
    return apply_plot_theme(fig)


def create_color_density_heatmap(dataset_df: pd.DataFrame) -> go.Figure:
    """Generates a density heatmap in the A-B plane."""
    fig = px.density_heatmap(
        dataset_df,
        x="A",
        y="B",
        nbinsx=50,
        nbinsy=50,
        title="Color Density in A-B Plane",
        labels={"A": "A Component", "B": "B Component"},
        color_continuous_scale=[THEME["bg_top"], THEME["accent_3"], THEME["accent"], THEME["accent_2"]],
        template="plotly_white",
    )
    fig.update_layout(xaxis_title="A", yaxis_title="B")
    return apply_plot_theme(fig)


@st.cache_data
def cached_lab_to_rgb(l: float, a: float, b: float) -> Tuple[int, int, int]:
    """Cached helper for converting LAB to RGB."""
    return lab_to_rgb([l, a, b])


def create_pairwise_scatter_matrix(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    closest_lab: List[float],
) -> go.Figure:
    """Creates a scatter matrix plot showing pairwise LAB relationships."""
    splom_df = dataset_df.copy()
    input_row = {"L": input_lab[0], "A": input_lab[1], "B": input_lab[2], "Color Name": "Input Color"}
    closest_row = {
        "L": closest_lab[0],
        "A": closest_lab[1],
        "B": closest_lab[2],
        "Color Name": "Closest Color",
    }
    splom_df = pd.concat([splom_df, pd.DataFrame([input_row, closest_row])], ignore_index=True)

    splom_df["Color Group"] = splom_df.apply(
        lambda row: format_rgb(cached_lab_to_rgb(row["L"], row["A"], row["B"])), axis=1
    )

    splom_trace = go.Splom(
        dimensions=[
            dict(label="L", values=splom_df["L"]),
            dict(label="A", values=splom_df["A"]),
            dict(label="B", values=splom_df["B"]),
        ],
        text=splom_df["Color Name"],
        marker=dict(size=5, color=splom_df["Color Group"], opacity=0.7),
        diagonal_visible=False,
        showupperhalf=False,
        name="Colors",
    )
    fig_splom = go.Figure(data=[splom_trace])
    fig_splom.update_layout(title="Pairwise LAB Relationships", dragmode="select", height=800)
    return apply_plot_theme(fig_splom)

# =============================================================================
# UI Helpers
# =============================================================================

def render_section_header(title: str, subtitle: Optional[str] = None) -> None:
    subtitle_html = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-title">{title}</div>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(input_rgb: Tuple[int, int, int], input_lab: List[float], method_label: str) -> None:
    swatch_color = format_rgb(input_rgb)
    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-body">
                <div class="hero-kicker">Color Intelligence Studio</div>
                <h1>Getty Colour Identifier</h1>
                <p>
                    Match LAB values against the ISCC-NBS archive with clarity and confidence.
                    Explore Delta-E precision, nuanced visual diagnostics, and Getty AAT term mapping.
                </p>
                <div class="hero-tags">
                    <span class="chip">LAB to RGB</span>
                    <span class="chip">Delta-E Analysis</span>
                    <span class="chip">AAT Term Mapping</span>
                </div>
            </div>
            <div class="hero-card">
                <div class="hero-label">Current Input</div>
                <div class="hero-swatch" style="background: {swatch_color};"></div>
                <div class="hero-meta">
                    <div>Method: {method_label}</div>
                    <div>LAB: L {input_lab[0]:.1f} | A {input_lab[1]:.1f} | B {input_lab[2]:.1f}</div>
                    <div>RGB: {rgb_to_hex(input_rgb)}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <h3>Upload your dataset to begin</h3>
            <p>Provide the ISCC-NBS LAB CSV in the sidebar to unlock the analysis suite.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(text: str) -> None:
    st.markdown(f"<div class='status-pill'>{text}</div>", unsafe_allow_html=True)


def render_color_card(title: str, subtitle: str, rgb: Tuple[int, int, int], lab: List[float]) -> None:
    st.markdown(
        f"""
        <div class="color-card">
            <div class="color-swatch" style="background: {format_rgb(rgb)}"></div>
            <div>
                <div class="color-card-title">{title}</div>
                <div class="color-card-subtitle">{subtitle}</div>
                <div class="color-card-hex">{rgb_to_hex(rgb)}</div>
                <div class="color-card-lab">L {lab[0]:.2f} | A {lab[1]:.2f} | B {lab[2]:.2f}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_blend_card(input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int]) -> None:
    gradient = f"linear-gradient(90deg, {format_rgb(input_rgb)}, {format_rgb(closest_rgb)})"
    st.markdown(
        f"""
        <div class="blend-card">
            <div class="blend-bar" style="background: {gradient};"></div>
            <div class="blend-meta">Input to Closest Match</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_results_dataframe(result: ColorMatchResult) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Metric": "Input LAB",
                "Value": f"L={result.input_lab[0]:.2f}, A={result.input_lab[1]:.2f}, B={result.input_lab[2]:.2f}",
            },
            {"Metric": "Closest ISCC-NBS Color", "Value": result.closest_name},
            {"Metric": f"Delta-E ({result.method_label})", "Value": result.delta_e_label},
            {
                "Metric": "Closest LAB",
                "Value": f"L={result.closest_lab[0]:.2f}, A={result.closest_lab[1]:.2f}, B={result.closest_lab[2]:.2f}",
            },
            {"Metric": "Input RGB", "Value": rgb_to_hex(result.input_rgb)},
            {"Metric": "Closest RGB", "Value": rgb_to_hex(result.closest_rgb)},
        ]
    )

# =============================================================================
# ColorAnalyzer Class
# =============================================================================

class ColorAnalyzer:
    """Encapsulates dataset loading, RDF processing, matching, and visualization."""

    REQUIRED_COLUMNS = REQUIRED_COLUMNS

    def __init__(self) -> None:
        self.dataset_df: Optional[pd.DataFrame] = None
        self.rdf_alternatives_df: Optional[pd.DataFrame] = None
        self.input_lab: Optional[List[float]] = None
        self.delta_e_method: str = "euclidean"
        self.result: Optional[ColorMatchResult] = None

    def load_dataset(self, uploaded_file: IO[Any]) -> bool:
        """Loads and validates the dataset CSV file."""
        try:
            self.dataset_df = load_dataset(uploaded_file)
        except DatasetError:
            return False
        return True

    def load_rdf(self, rdf_file: Optional[IO[Any]]) -> None:
        """Processes the RDF file and stores alternative color terms."""
        if rdf_file is not None:
            try:
                self.rdf_alternatives_df = extract_alternative_terms_rdf(rdf_file)
            except RDFParsingError:
                self.rdf_alternatives_df = None
        else:
            self.rdf_alternatives_df = None

    def set_input_color(self, lab: List[float]) -> bool:
        """Validates and sets the input LAB color."""
        if validate_lab_color(lab, report=True):
            self.input_lab = lab
            return True
        return False

    def set_delta_e_method(self, method: str) -> None:
        """Sets the Delta-E calculation method."""
        self.delta_e_method = method

    def match_color(self) -> Optional[ColorMatchResult]:
        """Finds the closest dataset color to the input LAB color."""
        if self.dataset_df is None or self.input_lab is None:
            _log_and_report("Dataset or input LAB color not set.", "error", "Processing Error")
            return None
        closest_color, delta_e = find_closest_color(self.input_lab, self.dataset_df, delta_e_method=self.delta_e_method)
        if closest_color is None or delta_e is None:
            return None
        closest_lab = [closest_color["L"], closest_color["A"], closest_color["B"]]
        result = ColorMatchResult(
            input_lab=self.input_lab,
            closest_lab=closest_lab,
            closest_name=closest_color["Color Name"],
            delta_e=float(delta_e),
            input_rgb=lab_to_rgb(self.input_lab),
            closest_rgb=lab_to_rgb(closest_lab),
            method=self.delta_e_method,
        )
        self.result = result
        return result

    def generate_visualizations(self, result: ColorMatchResult) -> Dict[str, go.Figure]:
        """Generates Plotly figures for the analysis suite."""
        if self.dataset_df is None:
            _log_and_report("Missing dataset for visualization.", "error", "Visualization Error")
            return {}
        figs = {
            "comparison": create_color_comparison_plot(
                result.input_rgb,
                result.closest_rgb,
                result.input_lab,
                result.closest_lab,
                result.closest_name,
                result.delta_e,
            ),
            "lab_bar": create_lab_comparison_bar(
                result.input_lab,
                result.closest_lab,
                result.closest_name,
                result.input_rgb,
                result.closest_rgb,
            ),
            "lab_3d": create_3d_lab_plot(
                result.input_lab,
                result.closest_lab,
                result.closest_name,
                self.dataset_df,
                result.input_rgb,
                result.closest_rgb,
            ),
            "delta_hist": create_delta_e_histogram(
                calculate_delta_e(result.input_lab, self.dataset_df, method=self.delta_e_method)
            ),
            "density_heatmap": create_color_density_heatmap(self.dataset_df),
            "scatter_matrix": create_pairwise_scatter_matrix(self.dataset_df, result.input_lab, result.closest_lab),
        }
        return figs

# =============================================================================
# Streamlit UI Functions
# =============================================================================

def display_sidebar() -> Tuple[Any, Any, List[float], str]:
    """Displays the sidebar widgets for file uploads and LAB input."""
    st.sidebar.header("Input Studio")
    st.sidebar.markdown("Upload your datasets and tune the LAB values.")
    csv_file = st.sidebar.file_uploader("ISCC-NBS LAB CSV", type=["csv"])
    rdf_file = st.sidebar.file_uploader("Getty AAT RDF (XML)", type=["xml"])
    with st.sidebar.expander("Workflow", expanded=True):
        st.markdown(
            """
            1. Upload the ISCC-NBS LAB dataset.
            2. Optionally upload the Getty AAT RDF file.
            3. Adjust the LAB values.
            4. Pick a Delta-E method.
            5. Analyze the closest color match.
            """
        )
    delta_e_metric = st.sidebar.radio("Delta-E method", ("Euclidean Delta E 76", "CIEDE2000"), index=0)
    st.sidebar.markdown("### LAB Input")
    input_method = st.sidebar.radio("Input mode", ("Slider", "Manual"), horizontal=True)
    if input_method == "Slider":
        lab_l = st.sidebar.slider("L", 0.0, 100.0, 50.0, 0.1)
        lab_a = st.sidebar.slider("A", -128.0, 127.0, 0.0, 0.1)
        lab_b = st.sidebar.slider("B", -128.0, 127.0, 0.0, 0.1)
    else:
        lab_l = st.sidebar.number_input("L (0-100)", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        lab_a = st.sidebar.number_input("A (-128 to 127)", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)
        lab_b = st.sidebar.number_input("B (-128 to 127)", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)
    input_lab = [lab_l, lab_a, lab_b]
    method_key = "ciede2000" if "CIEDE" in delta_e_metric else "euclidean"
    return csv_file, rdf_file, input_lab, method_key


def summarize_dataset(dataset_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "rows": len(dataset_df),
        "unique": dataset_df["Color Name"].nunique(),
        "l_min": dataset_df["L"].min(),
        "l_max": dataset_df["L"].max(),
        "a_min": dataset_df["A"].min(),
        "a_max": dataset_df["A"].max(),
        "b_min": dataset_df["B"].min(),
        "b_max": dataset_df["B"].max(),
    }


def render_dataset_overview(dataset_df: pd.DataFrame) -> None:
    summary = summarize_dataset(dataset_df)
    render_section_header("Dataset Overview", "A quick read on the ISCC-NBS archive.")
    cols = st.columns(4)
    cols[0].metric("Colors", f"{summary['rows']:,}")
    cols[1].metric("Unique Names", f"{summary['unique']:,}")
    cols[2].metric("L Range", f"{summary['l_min']:.1f} - {summary['l_max']:.1f}")
    cols[3].metric("A Range", f"{summary['a_min']:.1f} - {summary['a_max']:.1f}")
    cols = st.columns(4)
    cols[0].metric("B Range", f"{summary['b_min']:.1f} - {summary['b_max']:.1f}")
    cols[1].metric("Dataset Health", "Validated")
    cols[2].metric("Color Space", "CIELAB")
    cols[3].metric("Ready", "Yes")
    with st.expander("Preview dataset", expanded=False):
        st.dataframe(dataset_df.head(12), use_container_width=True)


def render_input_preview(input_lab: List[float], input_rgb: Tuple[int, int, int], method_label: str) -> None:
    render_section_header("Input Preview", "Dial in the LAB values before analysis.")
    cols = st.columns([1.2, 1])
    with cols[0]:
        render_color_card("Input LAB", method_label, input_rgb, input_lab)
    with cols[1]:
        st.markdown(
            """
            <div class="glass-card">
                <h3>Analysis Notes</h3>
                <p>Stay within valid LAB ranges for reliable conversion. Use CIEDE2000 for perceptual accuracy.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_results_section(result: ColorMatchResult, rdf_df: Optional[pd.DataFrame]) -> None:
    render_section_header("Match Studio", "Your closest ISCC-NBS match and key metrics.")
    cols = st.columns(3)
    cols[0].metric("Closest Match", result.closest_name)
    cols[1].metric("Delta-E", result.delta_e_label)
    cols[2].metric("Method", result.method_label)

    cards = st.columns(2)
    with cards[0]:
        render_color_card("Input Color", "Your LAB input", result.input_rgb, result.input_lab)
    with cards[1]:
        render_color_card("Closest Match", result.closest_name, result.closest_rgb, result.closest_lab)

    render_blend_card(result.input_rgb, result.closest_rgb)

    results_df = build_results_dataframe(result)
    with st.expander("Detailed results", expanded=False):
        st.dataframe(results_df, use_container_width=True)

    csv_results = results_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Results CSV",
        data=csv_results,
        file_name="color_match_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    if rdf_df is not None and not rdf_df.empty:
        render_section_header("Getty AAT Terms", "Alternative descriptions from the Getty AAT dataset.")
        cols = st.columns([1, 1.3])
        with cols[0]:
            st.dataframe(rdf_df, use_container_width=True)
        with cols[1]:
            fig_sunburst = create_alternative_terms_sunburst(rdf_df, base_color=format_rgb(result.closest_rgb))
            st.plotly_chart(fig_sunburst, use_container_width=True)


def render_visuals_section(figs: Dict[str, go.Figure]) -> None:
    render_section_header("Visual Diagnostics", "Explore structure, density, and distribution.")
    tabs = st.tabs(
        [
            "Comparison",
            "LAB Bars",
            "3D LAB",
            "Delta-E",
            "Density",
            "Scatter Matrix",
        ]
    )
    with tabs[0]:
        st.plotly_chart(figs["comparison"], use_container_width=True)
    with tabs[1]:
        st.plotly_chart(figs["lab_bar"], use_container_width=True)
    with tabs[2]:
        st.plotly_chart(figs["lab_3d"], use_container_width=True)
    with tabs[3]:
        st.plotly_chart(figs["delta_hist"], use_container_width=True)
    with tabs[4]:
        st.plotly_chart(figs["density_heatmap"], use_container_width=True)
    with tabs[5]:
        st.plotly_chart(figs["scatter_matrix"], use_container_width=True)

# =============================================================================
# Main App
# =============================================================================

def main() -> None:
    st.set_page_config(page_title="Getty Colour Identifier", layout="wide", page_icon="C")
    inject_global_styles()

    csv_file, rdf_file, input_lab, delta_e_method = display_sidebar()
    method_label = "CIEDE2000" if delta_e_method == "ciede2000" else "Euclidean Delta E 76"

    input_rgb = lab_to_rgb(input_lab) if validate_lab_color(input_lab, report=False) else (120, 120, 120)
    render_hero(input_rgb, input_lab, method_label)

    if csv_file is None:
        render_empty_state()
        return

    analyzer = ColorAnalyzer()
    with st.spinner("Loading dataset..."):
        if not analyzer.load_dataset(csv_file):
            st.error("Dataset loading failed. Please check the CSV file.")
            return

    render_status_pill("Dataset Ready")
    render_dataset_overview(analyzer.dataset_df)

    analyzer.load_rdf(rdf_file)
    analyzer.set_delta_e_method(delta_e_method)

    render_input_preview(input_lab, input_rgb, method_label)

    analyze = st.button("Analyze Color", type="primary", use_container_width=True)
    if analyze:
        if analyzer.set_input_color(input_lab):
            with st.spinner("Analyzing..."):
                result = analyzer.match_color()
            if result:
                render_results_section(result, analyzer.rdf_alternatives_df)
                figs = analyzer.generate_visualizations(result)
                if figs:
                    render_visuals_section(figs)
                else:
                    st.error("Error generating visualizations.")
            else:
                st.error("An error occurred during color matching. Please check your inputs and dataset.")
        else:
            st.error("Invalid LAB color input. Please check the values.")


if __name__ == "__main__":
    main()
