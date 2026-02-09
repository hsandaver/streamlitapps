"""
app.py

A refactored, design-forward Streamlit LAB Color Identifier.
Features modular data handling, refined visualizations, and a premium UI.
"""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from typing import Union, List, Tuple, Optional, Any, IO, Dict
import logging
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency fallback
    Image = None
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie1994, delta_e_cie2000, delta_e_cmc
from colormath.color_objects import LabColor, XYZColor, sRGBColor
from colormath.chromatic_adaptation import apply_chromatic_adaptation_on_color
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
LAB_COLUMN_ALIASES: Dict[str, Tuple[str, ...]] = {
    "L": ("L", "L*", "Lab L", "CIELAB L", "L Value", "Lightness"),
    "A": ("A", "A*", "Lab A", "CIELAB A", "A Value"),
    "B": ("B", "B*", "Lab B", "CIELAB B", "B Value"),
}
COLOR_NAME_COLUMN_ALIASES: Tuple[str, ...] = (
    "Color Name",
    "Colour Name",
    "Name",
    "Swatch Name",
    "Swatch",
    "Sample",
    "Label",
)
LAB_TRIPLET_COLUMN_ALIASES: Tuple[str, ...] = (
    "Lab",
    "CIELAB",
    "LAB Value",
    "LAB Values",
    "L*a*b*",
    "L*a*b",
)
NIX_MINI3_PROFILE_LABEL = "Nix Mini 3"

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
TAG_OPTIONS = ["Light", "Mid", "Deep", "Warm", "Cool", "Balanced", "Vivid", "Soft", "Muted"]
TAG_COLOR_PALETTE = [THEME["accent"], THEME["accent_2"], THEME["accent_3"]]
SPARKLINE_POINTS = 28

ILLUMINANT_OPTIONS = ["a", "b", "c", "d50", "d55", "d65", "d75", "e", "f2", "f7", "f11"]
OBSERVER_OPTIONS = ["2", "10"]
CHROMATIC_ADAPTATION_OPTIONS = ["bradford", "von_kries", "xyz_scaling", "none"]
CHROMATIC_ADAPTATION_LABELS = {
    "bradford": "Bradford",
    "von_kries": "Von Kries",
    "xyz_scaling": "XYZ Scaling",
    "none": "None",
}
CHROMATIC_ADAPTATION_UI_MAP = {
    "Bradford": "bradford",
    "Von Kries": "von_kries",
    "XYZ Scaling": "xyz_scaling",
    "None": "none",
}

DELTA_E_METHOD_OPTIONS = [
    "CIEDE2000",
    "CIE94",
    "CMC l:c",
    "Euclidean Delta E 76",
]
DELTA_E_METHOD_UI_MAP = {
    "CIEDE2000": "ciede2000",
    "CIE94": "cie94",
    "CMC l:c": "cmc",
    "Euclidean Delta E 76": "euclidean",
}
DELTA_E_METHOD_LABELS = {
    "ciede2000": "CIEDE2000",
    "cie94": "CIE94",
    "cmc": "CMC l:c",
    "euclidean": "Euclidean Delta E 76",
}
DELTA_E_PRESET_OPTIONS = ["Paint", "Textile"]
DELTA_E_PRESET_UI_MAP = {"Paint": "paint", "Textile": "textile"}
DELTA_E_PRESET_LABELS = {"paint": "Paint", "textile": "Textile"}
DELTA_E_PRESET_PROFILES = {
    "paint": {
        "cie94": {"K_L": 1, "K_C": 1, "K_H": 1, "K_1": 0.045, "K_2": 0.015},
        "cmc": {"pl": 1, "pc": 1},
    },
    "textile": {
        "cie94": {"K_L": 2, "K_C": 1, "K_H": 1, "K_1": 0.048, "K_2": 0.014},
        "cmc": {"pl": 2, "pc": 1},
    },
}
CONFIDENCE_TOP2_MARGIN_THRESHOLD = 0.5
CONFIDENCE_TOP3_MARGIN_THRESHOLD = 1.0
CONFIDENCE_RELATIVE_MARGIN_THRESHOLD = 0.15
LAB_L_RANGE = (0.0, 100.0)
LAB_A_RANGE = (-128.0, 127.0)
LAB_B_RANGE = (-128.0, 127.0)
QA_COVERAGE_BINS = 12
QA_MIN_COVERAGE_RATIO = 0.45
GAMUT_EPSILON = 1e-9
UNCERTAINTY_DEFAULT_SIGMA = 0.5
UNCERTAINTY_MIN_SIGMA = 0.0
UNCERTAINTY_MAX_SIGMA = 10.0
UNCERTAINTY_DEFAULT_SIMULATIONS = 300
UNCERTAINTY_MIN_SIMULATIONS = 100
UNCERTAINTY_MAX_SIMULATIONS = 2000
UNCERTAINTY_REPEAT_RATE_HIGH = 0.9
UNCERTAINTY_REPEAT_RATE_MODERATE = 0.75


def normalize_delta_e_preset(preset: str) -> str:
    normalized = (preset or "").lower()
    if normalized in DELTA_E_PRESET_PROFILES:
        return normalized
    return "paint"


def get_delta_e_method_label(method: str, preset: str = "paint") -> str:
    method_key = (method or "").lower()
    preset_key = normalize_delta_e_preset(preset)
    preset_label = DELTA_E_PRESET_LABELS[preset_key]
    if method_key == "cie94":
        return f"CIE94 ({preset_label})"
    if method_key == "cmc":
        cmc_profile = DELTA_E_PRESET_PROFILES[preset_key]["cmc"]
        return f"CMC l:c ({preset_label} {int(cmc_profile['pl'])}:{int(cmc_profile['pc'])})"
    return DELTA_E_METHOD_LABELS.get(method_key, DELTA_E_METHOD_LABELS["euclidean"])

# =============================================================================
# Data Models
# =============================================================================

@dataclass(frozen=True)
class RgbGamutDiagnostics:
    raw_rgb: Tuple[float, float, float]
    clipped_rgb: Tuple[int, int, int]
    clipped_channels: Tuple[str, ...]
    clip_magnitude: float

    @property
    def out_of_gamut(self) -> bool:
        return len(self.clipped_channels) > 0

    @property
    def status_label(self) -> str:
        return "Out of sRGB Gamut" if self.out_of_gamut else "In sRGB Gamut"

    @property
    def channels_label(self) -> str:
        return ", ".join(self.clipped_channels) if self.clipped_channels else "None"

    @property
    def warning_message(self) -> str:
        if not self.out_of_gamut:
            return "No clipping was required."
        return (
            f"Converted LAB is outside sRGB gamut; clipped channel(s): {self.channels_label} "
            f"(max channel excursion {self.clip_magnitude:.3f})."
        )


@dataclass
class ColorMatchResult:
    input_lab: List[float]
    closest_lab: List[float]
    closest_name: str
    delta_e: float
    input_rgb: Tuple[int, int, int]
    closest_rgb: Tuple[int, int, int]
    input_gamut: "RgbGamutDiagnostics"
    closest_gamut: "RgbGamutDiagnostics"
    method: str
    delta_e_preset: str
    confidence: "MatchConfidenceSummary"
    uncertainty_settings: "MeasurementUncertaintySettings"
    uncertainty: "MonteCarloStabilitySummary"
    science_settings: "ColorScienceSettings"

    @property
    def delta_e_label(self) -> str:
        return f"{self.delta_e:.2f}"

    @property
    def method_label(self) -> str:
        return get_delta_e_method_label(self.method, self.delta_e_preset)


@dataclass(frozen=True)
class MatchConfidenceSummary:
    best_delta_e: float
    second_delta_e: Optional[float]
    third_delta_e: Optional[float]
    margin_top2: Optional[float]
    margin_top3: Optional[float]
    confidence_score: float
    confidence_label: str
    is_ambiguous: bool
    ambiguity_reason: str
    competing_names: List[str]

    @property
    def score_label(self) -> str:
        return f"{self.confidence_score:.0f}/100 ({self.confidence_label})"


@dataclass(frozen=True)
class MeasurementUncertaintySettings:
    enabled: bool = False
    l_sigma: float = UNCERTAINTY_DEFAULT_SIGMA
    a_sigma: float = UNCERTAINTY_DEFAULT_SIGMA
    b_sigma: float = UNCERTAINTY_DEFAULT_SIGMA
    simulations: int = UNCERTAINTY_DEFAULT_SIMULATIONS
    random_seed: int = 42

    def normalized(self) -> "MeasurementUncertaintySettings":
        return MeasurementUncertaintySettings(
            enabled=bool(self.enabled),
            l_sigma=float(np.clip(float(self.l_sigma), UNCERTAINTY_MIN_SIGMA, UNCERTAINTY_MAX_SIGMA)),
            a_sigma=float(np.clip(float(self.a_sigma), UNCERTAINTY_MIN_SIGMA, UNCERTAINTY_MAX_SIGMA)),
            b_sigma=float(np.clip(float(self.b_sigma), UNCERTAINTY_MIN_SIGMA, UNCERTAINTY_MAX_SIGMA)),
            simulations=int(np.clip(int(self.simulations), UNCERTAINTY_MIN_SIMULATIONS, UNCERTAINTY_MAX_SIMULATIONS)),
            random_seed=max(0, int(self.random_seed)),
        )

    @property
    def sigma_vector(self) -> np.ndarray:
        return np.array([self.l_sigma, self.a_sigma, self.b_sigma], dtype=float)

    @property
    def sigma_label(self) -> str:
        return f"L {self.l_sigma:.2f}, A {self.a_sigma:.2f}, B {self.b_sigma:.2f}"


@dataclass(frozen=True)
class MonteCarloStabilitySummary:
    enabled: bool
    simulations_requested: int
    simulations_run: int
    top_match_repeat_rate: float
    stability_score: float
    stability_label: str
    mean_best_delta_e: Optional[float]
    std_best_delta_e: Optional[float]
    p05_best_delta_e: Optional[float]
    p95_best_delta_e: Optional[float]
    competitor_breakdown: List[Dict[str, Any]]
    is_unstable: bool
    message: str

    @property
    def repeat_rate_label(self) -> str:
        return f"{self.top_match_repeat_rate * 100:.1f}%"

    @property
    def stability_score_label(self) -> str:
        return f"{self.stability_score:.0f}/100 ({self.stability_label})"

    @property
    def delta_band_label(self) -> str:
        if self.mean_best_delta_e is None or self.std_best_delta_e is None:
            return "N/A"
        return f"{self.mean_best_delta_e:.2f} +/- {self.std_best_delta_e:.2f}"

    @property
    def percentile_band_label(self) -> str:
        if self.p05_best_delta_e is None or self.p95_best_delta_e is None:
            return "N/A"
        return f"{self.p05_best_delta_e:.2f} to {self.p95_best_delta_e:.2f}"


@dataclass(frozen=True)
class DatasetQualityReport:
    rows_original: int
    rows_final: int
    invalid_rows_removed: int
    out_of_range_rows_removed: int
    duplicate_rows_removed: int
    outlier_rows: int
    l_coverage_ratio: float
    a_coverage_ratio: float
    b_coverage_ratio: float
    sparse_axes: Tuple[str, ...]

    @property
    def rows_removed_total(self) -> int:
        return self.invalid_rows_removed + self.out_of_range_rows_removed + self.duplicate_rows_removed

    @property
    def coverage_summary_label(self) -> str:
        return f"L {self.l_coverage_ratio * 100:.0f}% / A {self.a_coverage_ratio * 100:.0f}% / B {self.b_coverage_ratio * 100:.0f}%"

    @property
    def health_label(self) -> str:
        if self.rows_removed_total == 0 and self.outlier_rows == 0 and not self.sparse_axes:
            return "Strong"
        if self.rows_removed_total <= max(2, int(self.rows_original * 0.02)) and len(self.sparse_axes) <= 1:
            return "Acceptable"
        return "Needs Review"

    @property
    def warnings(self) -> List[str]:
        messages: List[str] = []
        if self.invalid_rows_removed > 0:
            messages.append(f"Removed {self.invalid_rows_removed} rows with missing or non-numeric required values.")
        if self.out_of_range_rows_removed > 0:
            messages.append(
                f"Removed {self.out_of_range_rows_removed} rows outside LAB ranges "
                f"(L {int(LAB_L_RANGE[0])}-{int(LAB_L_RANGE[1])}, "
                f"A {int(LAB_A_RANGE[0])}-{int(LAB_A_RANGE[1])}, "
                f"B {int(LAB_B_RANGE[0])}-{int(LAB_B_RANGE[1])})."
            )
        if self.duplicate_rows_removed > 0:
            messages.append(f"Removed {self.duplicate_rows_removed} duplicate LAB+name rows.")
        if self.outlier_rows > 0:
            messages.append(f"Flagged {self.outlier_rows} potential outlier rows (IQR rule).")
        if self.sparse_axes:
            axis_text = ", ".join(self.sparse_axes)
            messages.append(
                f"Coverage gaps detected on axis/axes: {axis_text} (coverage below {int(QA_MIN_COVERAGE_RATIO * 100)}%)."
            )
        return messages


@dataclass(frozen=True)
class ColorScienceSettings:
    source_illuminant: str = "d50"
    target_illuminant: str = "d65"
    observer: str = "2"
    chromatic_adaptation: str = "bradford"

    def normalized(self) -> "ColorScienceSettings":
        source = self.source_illuminant.lower()
        target = self.target_illuminant.lower()
        observer = str(self.observer)
        adaptation = self.chromatic_adaptation.lower()
        if source not in ILLUMINANT_OPTIONS:
            source = "d50"
        if target not in ILLUMINANT_OPTIONS:
            target = "d65"
        if observer not in OBSERVER_OPTIONS:
            observer = "2"
        if adaptation not in CHROMATIC_ADAPTATION_OPTIONS:
            adaptation = "bradford"
        return ColorScienceSettings(
            source_illuminant=source,
            target_illuminant=target,
            observer=observer,
            chromatic_adaptation=adaptation,
        )

    @property
    def source_label(self) -> str:
        return self.source_illuminant.upper()

    @property
    def target_label(self) -> str:
        return self.target_illuminant.upper()

    @property
    def observer_label(self) -> str:
        return f"{self.observer} deg"

    @property
    def adaptation_label(self) -> str:
        return CHROMATIC_ADAPTATION_LABELS.get(self.chromatic_adaptation, self.chromatic_adaptation)

    @property
    def adaptation_active(self) -> bool:
        return self.source_illuminant != self.target_illuminant and self.chromatic_adaptation != "none"

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

        span[data-testid="stIconMaterial"] {{
            font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Symbols Sharp", "Material Icons", sans-serif !important;
            font-variation-settings: "FILL" 0, "wght" 400, "GRAD" 0, "opsz" 24;
            line-height: 1;
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
            box-shadow: 0 14px 32px rgba(18, 20, 26, 0.12);
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

        .section-accent {{
            width: 92px;
            height: 4px;
            border-radius: 999px;
            margin-top: 0.6rem;
            background: linear-gradient(90deg, var(--accent), var(--accent-3));
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
            min-height: 170px;
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
            min-height: 170px;
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

        .section-spacer {{
            height: 16px;
        }}

        .section-spacer-lg {{
            height: 28px;
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

        .status-wrap {{
            margin-top: 0.6rem;
            margin-bottom: 0.35rem;
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

        div[data-testid="stMetric"] svg {{
            opacity: 0.35;
        }}

        div[data-testid="stFileUploader"] section {{
            padding: 0.55rem 0.8rem !important;
        }}

        .sidebar-divider {{
            height: 1px;
            background: rgba(24, 24, 28, 0.08);
            margin: 0.75rem 0 1rem;
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
            background: linear-gradient(180deg, rgba(255, 255, 255, 0.96), rgba(247, 248, 251, 0.98));
            color: var(--ink);
            border-right: 1px solid rgba(24, 24, 28, 0.08);
        }}

        div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {{
            color: var(--ink);
        }}

        div[data-testid="stSidebar"] label, div[data-testid="stSidebar"] span {{
            color: var(--muted);
        }}

        /* Keep selectbox dropdown content readable even when rendered in portals. */
        div[data-baseweb="popover"] [role="listbox"] {{
            background: rgba(255, 255, 255, 0.98) !important;
            border: 1px solid var(--card-border) !important;
            border-radius: 12px !important;
        }}

        div[data-baseweb="popover"] [role="option"] {{
            color: var(--ink) !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}

        div[data-baseweb="popover"] [role="option"] * {{
            color: inherit !important;
            opacity: 1 !important;
            visibility: visible !important;
        }}

        .sidebar-card {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(24, 24, 28, 0.12);
            border-radius: 16px;
            padding: 1rem 1.1rem;
            margin: 0.75rem 0 1rem;
            color: var(--ink);
            box-shadow: 0 12px 28px rgba(18, 20, 26, 0.06);
        }}

        .sidebar-card h4 {{
            margin: 0 0 0.5rem;
            font-size: 0.95rem;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--muted);
        }}

        .sidebar-card ol {{
            margin: 0.2rem 0 0 1rem;
            padding: 0;
            line-height: 1.6;
            font-size: 0.85rem;
            color: var(--muted);
        }}

        div[data-testid="stWidgetHoverOverlay"],
        div[data-testid="stWidgetLabelDetails"],
        div[data-testid="stElementToolbar"],
        div[data-testid="stElementToolbar"] *,
        [title="key"],
        [aria-label="key"] {{
            display: none !important;
            visibility: hidden !important;
        }}

        kbd {{
            display: none !important;
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


def suppress_key_overlay() -> None:
    """Last-resort suppression for stray Streamlit hover/tool overlays."""
    components.html(
        """
        <script>
        (function () {
          const matchesKey = (node) => {
            if (!node || !node.textContent) return false;
            const text = node.textContent.trim().toLowerCase();
            if (!text) return false;
            if (text.length > 12) return false;
            return text === "key" || text === "keyb" || text === "keyboard";
          };

          const looksLikeOverlay = (node) => {
            try {
              const style = window.getComputedStyle(node);
              if (style.position !== "fixed" && style.position !== "absolute") return false;
              const rect = node.getBoundingClientRect();
              return rect.top >= 0 && rect.top < 90 && rect.left < 220 && rect.width < 180;
            } catch (err) {
              return false;
            }
          };

          const hideIfMatch = (node) => {
            if (!node || node.nodeType !== 1) return;
            if (matchesKey(node) && looksLikeOverlay(node)) {
              node.style.display = "none";
              node.style.visibility = "hidden";
              node.setAttribute("data-hidden-by", "suppress_key_overlay");
            }
          };

          const scan = () => {
            document.querySelectorAll("body *").forEach((node) => hideIfMatch(node));
          };

          scan();
          const observer = new MutationObserver((mutations) => {
            for (const mutation of mutations) {
              mutation.addedNodes.forEach((node) => {
                hideIfMatch(node);
                if (node.querySelectorAll) {
                  node.querySelectorAll("*").forEach((child) => hideIfMatch(child));
                }
              });
            }
          });
          observer.observe(document.body, { childList: true, subtree: true });
        })();
        </script>
        """,
        height=0,
        width=0,
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


def normalize_column_key(column_name: Any) -> str:
    """Normalizes a CSV column name for alias matching."""
    return re.sub(r"[^a-z0-9]+", "", str(column_name).strip().lower())


def find_first_matching_column(columns: List[str], aliases: Tuple[str, ...]) -> Optional[str]:
    normalized_lookup: Dict[str, str] = {}
    for col in columns:
        normalized_lookup.setdefault(normalize_column_key(col), col)
    for alias in aliases:
        normalized_alias = normalize_column_key(alias)
        if normalized_alias in normalized_lookup:
            return normalized_lookup[normalized_alias]
    return None


def coerce_lab_series(series: pd.Series) -> pd.Series:
    """Converts mixed-format LAB text/number series to numeric values."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    normalized = series.astype(str).str.strip().str.replace(",", ".", regex=False)
    numeric = pd.to_numeric(normalized, errors="coerce")
    unresolved_mask = numeric.isna() & normalized.ne("")
    if unresolved_mask.any():
        extracted = normalized[unresolved_mask].str.extract(
            r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
            expand=False,
        )
        numeric.loc[unresolved_mask] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def parse_lab_triplet_text(raw_value: Any) -> Optional[Tuple[float, float, float]]:
    """Parses textual LAB forms like 'L* 50, a* -3, b* 12' or '50,-3,12'."""
    if raw_value is None:
        return None
    text = str(raw_value).strip()
    if not text:
        return None
    numeric_pattern = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"
    l_match = re.search(rf"(?i)\bl\*?\b\s*[:=]?\s*({numeric_pattern})", text)
    a_match = re.search(rf"(?i)\ba\*?\b\s*[:=]?\s*({numeric_pattern})", text)
    b_match = re.search(rf"(?i)\bb\*?\b\s*[:=]?\s*({numeric_pattern})", text)
    if l_match and a_match and b_match:
        return (float(l_match.group(1)), float(a_match.group(1)), float(b_match.group(1)))

    normalized_text = text.replace(";", " ").replace("|", " ").replace("/", " ")
    numeric_tokens = re.findall(numeric_pattern, normalized_text)
    if len(numeric_tokens) >= 3:
        return (float(numeric_tokens[0]), float(numeric_tokens[1]), float(numeric_tokens[2]))
    return None


def parse_pasted_lab_measurements(raw_text: str) -> pd.DataFrame:
    """Parses multiline pasted LAB rows into a standard measurement dataframe."""
    rows: List[Dict[str, Any]] = []
    for line in raw_text.splitlines():
        parsed = parse_lab_triplet_text(line)
        if parsed is None:
            continue
        rows.append(
            {
                "Color Name": f"Scan {len(rows) + 1}",
                "L": float(parsed[0]),
                "A": float(parsed[1]),
                "B": float(parsed[2]),
            }
        )
    return pd.DataFrame(rows, columns=["Color Name", "L", "A", "B"])


def aggregate_lab_measurements(measurements_df: pd.DataFrame, aggregation: str = "median") -> List[float]:
    """Aggregates multiple LAB scans into one representative LAB triplet."""
    if measurements_df.empty:
        return [50.0, 0.0, 0.0]
    lab_values = measurements_df[["L", "A", "B"]].astype(float)
    agg_key = aggregation.lower()
    if agg_key == "latest":
        selected = lab_values.iloc[-1]
    elif agg_key == "mean":
        selected = lab_values.mean(axis=0)
    else:
        selected = lab_values.median(axis=0)
    return [float(selected["L"]), float(selected["A"]), float(selected["B"])]


def clean_lab_measurements(measurements_df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Filters invalid or out-of-range LAB measurement rows."""
    if measurements_df.empty:
        return measurements_df, 0
    cleaned = measurements_df.copy()
    for col in ["L", "A", "B"]:
        cleaned[col] = coerce_lab_series(cleaned[col])

    valid_mask = (
        cleaned["L"].between(LAB_L_RANGE[0], LAB_L_RANGE[1], inclusive="both")
        & cleaned["A"].between(LAB_A_RANGE[0], LAB_A_RANGE[1], inclusive="both")
        & cleaned["B"].between(LAB_B_RANGE[0], LAB_B_RANGE[1], inclusive="both")
    )
    valid_mask = valid_mask & cleaned[["L", "A", "B"]].notnull().all(axis=1)
    dropped_rows = int((~valid_mask).sum())
    cleaned = cleaned.loc[valid_mask].reset_index(drop=True)
    return cleaned, dropped_rows


def load_lab_measurements_csv(uploaded_file: IO[Any], default_name_prefix: str = "Scan") -> Tuple[pd.DataFrame, Dict[str, str], int]:
    """
    Loads measurement rows from flexible CSV formats (including Nix exports).
    Returns cleaned LAB rows, detected column mapping, and dropped row count.
    """
    raw_df = read_uploaded_csv_flexible(uploaded_file)
    standardized_df, column_map = standardize_lab_dataframe(raw_df, default_name_prefix=default_name_prefix)
    cleaned_df, dropped_rows = clean_lab_measurements(standardized_df)
    return cleaned_df, column_map, dropped_rows


def read_uploaded_csv_flexible(uploaded_file: IO[Any]) -> pd.DataFrame:
    """
    Reads CSV uploads from mixed tooling exports (including semicolon-delimited files).
    """
    try:
        data_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    except Exception as exc:
        raise DatasetError(f"Unable to read uploaded file: {exc}") from exc
    if data_bytes is None:
        raise DatasetError("Uploaded file is empty.")
    if isinstance(data_bytes, str):
        data_bytes = data_bytes.encode("utf-8")

    attempts = (
        {},
        {"sep": None, "engine": "python"},
        {"sep": ";"},
    )
    fallback_df: Optional[pd.DataFrame] = None
    last_exc: Optional[Exception] = None
    for kwargs in attempts:
        try:
            candidate = pd.read_csv(io.BytesIO(data_bytes), **kwargs)
        except Exception as exc:
            last_exc = exc
            continue
        if candidate.shape[1] > 1:
            return candidate
        if fallback_df is None:
            fallback_df = candidate
    if fallback_df is not None:
        return fallback_df
    if last_exc is not None:
        raise DatasetError(f"Unable to parse CSV file: {last_exc}") from last_exc
    raise DatasetError("Unable to parse CSV file.")


def standardize_lab_dataframe(
    dataframe: pd.DataFrame,
    default_name_prefix: str = "Color",
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Maps flexible LAB/name column variants onto canonical columns:
    L, A, B, Color Name.
    """
    if dataframe is None:
        raise DatasetError("No dataframe provided.")
    working_df = dataframe.copy()
    source_columns = list(working_df.columns)
    column_map: Dict[str, str] = {}

    for axis in ("L", "A", "B"):
        axis_aliases = LAB_COLUMN_ALIASES.get(axis, tuple()) + (axis,)
        matched_col = find_first_matching_column(source_columns, axis_aliases)
        if matched_col is not None:
            column_map[axis] = matched_col

    if len(column_map) < 3:
        triplet_col = find_first_matching_column(source_columns, LAB_TRIPLET_COLUMN_ALIASES)
        if triplet_col is not None:
            parsed_triplets = working_df[triplet_col].apply(parse_lab_triplet_text)
            parsed_df = pd.DataFrame(parsed_triplets.tolist(), columns=["L", "A", "B"], index=working_df.index)
            for axis in ("L", "A", "B"):
                if axis not in column_map:
                    parsed_col_name = f"__parsed_{axis}"
                    working_df[parsed_col_name] = parsed_df[axis]
                    column_map[axis] = parsed_col_name

    missing_axes = [axis for axis in ("L", "A", "B") if axis not in column_map]
    if missing_axes:
        raise DatasetError(
            "Could not detect required LAB columns. "
            f"Missing: {', '.join(missing_axes)}."
        )

    color_name_col = find_first_matching_column(source_columns, COLOR_NAME_COLUMN_ALIASES)
    if color_name_col:
        color_names = working_df[color_name_col].astype(str).str.strip()
        column_map["Color Name"] = color_name_col
    else:
        color_names = pd.Series(
            [f"{default_name_prefix} {idx + 1}" for idx in range(len(working_df))],
            index=working_df.index,
        )
        column_map["Color Name"] = "<generated>"

    blank_name_mask = color_names.str.strip().eq("") | color_names.str.lower().eq("nan")
    if blank_name_mask.any():
        blank_positions = np.flatnonzero(blank_name_mask.to_numpy())
        for pos in blank_positions:
            row_idx = color_names.index[pos]
            color_names.loc[row_idx] = f"{default_name_prefix} {int(pos) + 1}"

    standardized = pd.DataFrame(
        {
            "L": coerce_lab_series(working_df[column_map["L"]]),
            "A": coerce_lab_series(working_df[column_map["A"]]),
            "B": coerce_lab_series(working_df[column_map["B"]]),
            "Color Name": color_names,
        }
    )
    return standardized, column_map


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
def transform_lab_triplet(
    l_value: float,
    a_value: float,
    b_value: float,
    source_illuminant: str = "d50",
    target_illuminant: str = "d65",
    observer: str = "2",
    chromatic_adaptation: str = "bradford",
) -> Tuple[float, float, float]:
    """
    Transforms a LAB triplet between illuminants using the selected observer and
    chromatic adaptation method.
    """
    source = source_illuminant.lower()
    target = target_illuminant.lower()
    observer_norm = str(observer)
    adaptation = chromatic_adaptation.lower()

    lab_color = LabColor(
        lab_l=float(l_value),
        lab_a=float(a_value),
        lab_b=float(b_value),
        observer=observer_norm,
        illuminant=source,
    )
    if source == target or adaptation == "none":
        return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)

    xyz_color = convert_color(lab_color, XYZColor)
    adapted_xyz = apply_chromatic_adaptation_on_color(xyz_color, target, adaptation=adaptation)
    adapted_lab = convert_color(adapted_xyz, LabColor, target_illuminant=target)
    return (adapted_lab.lab_l, adapted_lab.lab_a, adapted_lab.lab_b)


def transform_lab_array(
    lab_values: np.ndarray,
    settings: ColorScienceSettings,
) -> np.ndarray:
    """
    Applies illuminant adaptation to a LAB matrix if required.
    Returns the original array if adaptation is not active.
    """
    settings = settings.normalized()
    if not settings.adaptation_active:
        return lab_values.astype(float)
    try:
        transformed_rows = [
            transform_lab_triplet(
                row[0],
                row[1],
                row[2],
                settings.source_illuminant,
                settings.target_illuminant,
                settings.observer,
                settings.chromatic_adaptation,
            )
            for row in lab_values
        ]
        return np.array(transformed_rows, dtype=float)
    except Exception as exc:
        _log_and_report(
            f"Chromatic adaptation failed; using source LAB values. {exc}",
            "warning",
            "Conversion Warning",
        )
        return lab_values.astype(float)


def build_rgb_gamut_diagnostics(raw_rgb: Tuple[float, float, float]) -> RgbGamutDiagnostics:
    clipped_channels: List[str] = []
    clip_deltas: List[float] = []
    clipped_values: List[int] = []

    for channel_name, value in zip(("R", "G", "B"), raw_rgb):
        clipped_unit = float(np.clip(value, 0.0, 1.0))
        clipped_values.append(int(clipped_unit * 255))
        delta = abs(clipped_unit - float(value))
        clip_deltas.append(delta)
        if delta > GAMUT_EPSILON:
            clipped_channels.append(channel_name)

    clip_magnitude = max(clip_deltas) if clip_deltas else 0.0
    return RgbGamutDiagnostics(
        raw_rgb=(float(raw_rgb[0]), float(raw_rgb[1]), float(raw_rgb[2])),
        clipped_rgb=(clipped_values[0], clipped_values[1], clipped_values[2]),
        clipped_channels=tuple(clipped_channels),
        clip_magnitude=float(clip_magnitude),
    )


@st.cache_data(show_spinner=False)
def lab_to_rgb_with_diagnostics(
    lab_color: Union[List[float], Tuple[float, float, float], np.ndarray],
    source_illuminant: str = "d50",
    target_illuminant: str = "d65",
    observer: str = "2",
    chromatic_adaptation: str = "bradford",
) -> Tuple[Tuple[int, int, int], RgbGamutDiagnostics]:
    """
    Converts LAB to RGB and returns clipping diagnostics for sRGB gamut checks.
    """
    try:
        source = source_illuminant.lower()
        target = target_illuminant.lower()
        adaptation = chromatic_adaptation.lower()
        transformed = transform_lab_triplet(
            float(lab_color[0]),
            float(lab_color[1]),
            float(lab_color[2]),
            source_illuminant=source,
            target_illuminant=target,
            observer=observer,
            chromatic_adaptation=adaptation,
        )
        conversion_illuminant = source if adaptation == "none" else target
        rgb_lab = LabColor(
            lab_l=transformed[0],
            lab_a=transformed[1],
            lab_b=transformed[2],
            observer=str(observer),
            illuminant=conversion_illuminant,
        )
        rgb = convert_color(rgb_lab, sRGBColor, target_illuminant=conversion_illuminant)
        diagnostics = build_rgb_gamut_diagnostics((rgb.rgb_r, rgb.rgb_g, rgb.rgb_b))
        return diagnostics.clipped_rgb, diagnostics
    except Exception as exc:
        _log_and_report(f"Error converting LAB to RGB: {exc}", "error", "Conversion Error")
        fallback = build_rgb_gamut_diagnostics((0.0, 0.0, 0.0))
        return (0, 0, 0), fallback


@st.cache_data(show_spinner=False)
def lab_to_rgb(
    lab_color: Union[List[float], Tuple[float, float, float], np.ndarray],
    source_illuminant: str = "d50",
    target_illuminant: str = "d65",
    observer: str = "2",
    chromatic_adaptation: str = "bradford",
) -> Tuple[int, int, int]:
    """
    Converts a LAB color to RGB using colormath.
    Returns an RGB tuple with values between 0 and 255.
    """
    rgb, _ = lab_to_rgb_with_diagnostics(
        lab_color,
        source_illuminant=source_illuminant,
        target_illuminant=target_illuminant,
        observer=observer,
        chromatic_adaptation=chromatic_adaptation,
    )
    return rgb


def format_rgb(rgb: Tuple[int, int, int]) -> str:
    """Formats an RGB tuple into a Plotly-friendly RGB string."""
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Converts an RGB tuple into a hex string."""
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def parse_hex_color(hex_value: str) -> Optional[Tuple[int, int, int]]:
    """Parses #RRGGBB or #RGB hex strings into an RGB tuple."""
    if not isinstance(hex_value, str):
        return None
    cleaned = hex_value.strip()
    if cleaned.startswith("#"):
        cleaned = cleaned[1:]
    if len(cleaned) == 3 and re.fullmatch(r"[0-9a-fA-F]{3}", cleaned):
        cleaned = "".join(ch * 2 for ch in cleaned)
    if len(cleaned) != 6 or re.fullmatch(r"[0-9a-fA-F]{6}", cleaned) is None:
        return None
    return tuple(int(cleaned[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_lab(
    rgb: Tuple[int, int, int],
    source_illuminant: str = "d50",
    observer: str = "2",
) -> List[float]:
    """Converts an 8-bit RGB tuple to LAB for the selected illuminant/observer."""
    try:
        rgb_color = sRGBColor(rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0)
        xyz = convert_color(rgb_color, XYZColor)
        xyz_for_observer = XYZColor(
            xyz.xyz_x,
            xyz.xyz_y,
            xyz.xyz_z,
            observer=str(observer),
            illuminant=xyz.illuminant,
        )
        lab = convert_color(xyz_for_observer, LabColor, target_illuminant=source_illuminant.lower())
        return [float(lab.lab_l), float(lab.lab_a), float(lab.lab_b)]
    except Exception as exc:
        _log_and_report(f"Error converting RGB to LAB: {exc}", "error", "Conversion Error")
        return [50.0, 0.0, 0.0]


def calculate_delta_e(
    input_lab: Union[List[float], Tuple[float, float, float]],
    dataset_df: pd.DataFrame,
    method: str = "euclidean",
    delta_e_preset: str = "paint",
    science_settings: Optional[ColorScienceSettings] = None,
) -> np.ndarray:
    """
    Calculates Delta-E values between input LAB and dataset colors.
    Supports Euclidean (Delta E 76), CIEDE2000, CIE94, and CMC l:c.
    """
    settings = (science_settings or ColorScienceSettings()).normalized()
    method_key = (method or "").lower()
    preset_key = normalize_delta_e_preset(delta_e_preset)
    input_lab_arr = np.array(input_lab, dtype=float)
    dataset_lab_arr = dataset_df[["L", "A", "B"]].values.astype(float)

    if settings.adaptation_active:
        input_lab_arr = np.array(
            transform_lab_triplet(
                input_lab_arr[0],
                input_lab_arr[1],
                input_lab_arr[2],
                settings.source_illuminant,
                settings.target_illuminant,
                settings.observer,
                settings.chromatic_adaptation,
            ),
            dtype=float,
        )
        dataset_lab_arr = transform_lab_array(dataset_lab_arr, settings)

    analysis_illuminant = settings.target_illuminant if settings.adaptation_active else settings.source_illuminant

    if method_key == "euclidean":
        delta_e_values = np.linalg.norm(dataset_lab_arr - input_lab_arr, axis=1)
        return delta_e_values

    input_lab_obj = LabColor(
        lab_l=float(input_lab_arr[0]),
        lab_a=float(input_lab_arr[1]),
        lab_b=float(input_lab_arr[2]),
        observer=settings.observer,
        illuminant=analysis_illuminant,
    )
    comparison_lab_objects = [
        LabColor(
            lab_l=float(row[0]),
            lab_a=float(row[1]),
            lab_b=float(row[2]),
            observer=settings.observer,
            illuminant=analysis_illuminant,
        )
        for row in dataset_lab_arr
    ]

    if method_key == "ciede2000":
        return np.array([delta_e_cie2000(input_lab_obj, candidate) for candidate in comparison_lab_objects])
    if method_key == "cie94":
        cie94_profile = DELTA_E_PRESET_PROFILES[preset_key]["cie94"]
        return np.array(
            [delta_e_cie1994(input_lab_obj, candidate, **cie94_profile) for candidate in comparison_lab_objects]
        )
    if method_key == "cmc":
        cmc_profile = DELTA_E_PRESET_PROFILES[preset_key]["cmc"]
        return np.array([delta_e_cmc(input_lab_obj, candidate, **cmc_profile) for candidate in comparison_lab_objects])

    _log_and_report(
        f"Unknown Delta-E method '{method_key}', defaulting to Euclidean Delta E 76.",
        "warning",
        "Input Warning",
    )
    return np.linalg.norm(dataset_lab_arr - input_lab_arr, axis=1)


def build_ranked_matches(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    method: str,
    delta_e_preset: str,
    science_settings: ColorScienceSettings,
) -> pd.DataFrame:
    """Builds a full ranked table sorted by Delta-E distance."""
    delta_e_values = calculate_delta_e(
        input_lab,
        dataset_df,
        method=method,
        delta_e_preset=delta_e_preset,
        science_settings=science_settings,
    )
    ranked = dataset_df.copy()
    ranked["Delta E"] = delta_e_values
    ranked = ranked.dropna(subset=["Delta E"]).sort_values("Delta E").reset_index(drop=True)
    return ranked


def confidence_label_from_score(score: float) -> str:
    if score >= 80:
        return "High"
    if score >= 55:
        return "Moderate"
    return "Low"


def compute_match_confidence_from_ranked(ranked: pd.DataFrame) -> MatchConfidenceSummary:
    """Computes match confidence from a sorted Delta-E table."""
    if ranked.empty:
        return MatchConfidenceSummary(
            best_delta_e=float("nan"),
            second_delta_e=None,
            third_delta_e=None,
            margin_top2=None,
            margin_top3=None,
            confidence_score=0.0,
            confidence_label="Low",
            is_ambiguous=True,
            ambiguity_reason="No valid candidates were available for confidence estimation.",
            competing_names=[],
        )

    top = ranked.head(3).copy()
    best = float(top.iloc[0]["Delta E"])
    second = float(top.iloc[1]["Delta E"]) if len(top) >= 2 else None
    third = float(top.iloc[2]["Delta E"]) if len(top) >= 3 else None
    margin_top2 = (second - best) if second is not None else None
    margin_top3 = (third - best) if third is not None else None
    relative_top2_margin = (margin_top2 / max(best, 1e-6)) if margin_top2 is not None else None

    quality_component = float(np.clip(1 - (best / 8.0), 0.0, 1.0))
    separation_component = float(np.clip((margin_top2 or 0.0) / 2.0, 0.0, 1.0))
    cluster_component = float(np.clip((margin_top3 or margin_top2 or 0.0) / 3.0, 0.0, 1.0))
    confidence_score = round(100 * (0.55 * quality_component + 0.30 * separation_component + 0.15 * cluster_component), 1)
    confidence_label = confidence_label_from_score(confidence_score)

    ambiguity_reason = "Top match is clearly separated from nearest alternatives."
    is_ambiguous = False
    if second is None:
        is_ambiguous = True
        ambiguity_reason = "Only one candidate is available, so ambiguity cannot be ruled out."
    elif margin_top2 is not None and margin_top2 < CONFIDENCE_TOP2_MARGIN_THRESHOLD:
        is_ambiguous = True
        ambiguity_reason = f"Top two matches are very close (Delta-E margin < {CONFIDENCE_TOP2_MARGIN_THRESHOLD:.1f})."
    elif relative_top2_margin is not None and relative_top2_margin < CONFIDENCE_RELATIVE_MARGIN_THRESHOLD:
        is_ambiguous = True
        ambiguity_reason = f"Top-two separation is below {int(CONFIDENCE_RELATIVE_MARGIN_THRESHOLD * 100)}% of the best match."
    elif margin_top3 is not None and margin_top3 < CONFIDENCE_TOP3_MARGIN_THRESHOLD:
        is_ambiguous = True
        ambiguity_reason = f"Top three candidates cluster within Delta-E < {CONFIDENCE_TOP3_MARGIN_THRESHOLD:.1f}."

    competing_names = top["Color Name"].astype(str).tolist()
    return MatchConfidenceSummary(
        best_delta_e=best,
        second_delta_e=second,
        third_delta_e=third,
        margin_top2=margin_top2,
        margin_top3=margin_top3,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        is_ambiguous=is_ambiguous,
        ambiguity_reason=ambiguity_reason,
        competing_names=competing_names,
    )


def compute_match_confidence(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    method: str,
    delta_e_preset: str,
    science_settings: ColorScienceSettings,
) -> MatchConfidenceSummary:
    ranked = build_ranked_matches(
        dataset_df,
        input_lab,
        method=method,
        delta_e_preset=delta_e_preset,
        science_settings=science_settings,
    )
    return compute_match_confidence_from_ranked(ranked)


def stability_label_from_repeat_rate(repeat_rate: float) -> str:
    if repeat_rate >= UNCERTAINTY_REPEAT_RATE_HIGH:
        return "High"
    if repeat_rate >= UNCERTAINTY_REPEAT_RATE_MODERATE:
        return "Moderate"
    return "Low"


def build_disabled_uncertainty_summary(settings: MeasurementUncertaintySettings) -> MonteCarloStabilitySummary:
    return MonteCarloStabilitySummary(
        enabled=False,
        simulations_requested=settings.simulations,
        simulations_run=0,
        top_match_repeat_rate=0.0,
        stability_score=0.0,
        stability_label="N/A",
        mean_best_delta_e=None,
        std_best_delta_e=None,
        p05_best_delta_e=None,
        p95_best_delta_e=None,
        competitor_breakdown=[],
        is_unstable=False,
        message="Monte Carlo stability is disabled.",
    )


def compute_uncertainty_stability(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    baseline_match_name: str,
    method: str,
    delta_e_preset: str,
    science_settings: ColorScienceSettings,
    uncertainty_settings: MeasurementUncertaintySettings,
) -> MonteCarloStabilitySummary:
    settings = uncertainty_settings.normalized()
    if not settings.enabled:
        return build_disabled_uncertainty_summary(settings)
    if dataset_df.empty:
        return MonteCarloStabilitySummary(
            enabled=True,
            simulations_requested=settings.simulations,
            simulations_run=0,
            top_match_repeat_rate=0.0,
            stability_score=0.0,
            stability_label="Low",
            mean_best_delta_e=None,
            std_best_delta_e=None,
            p05_best_delta_e=None,
            p95_best_delta_e=None,
            competitor_breakdown=[],
            is_unstable=True,
            message="Dataset is empty; uncertainty simulation could not run.",
        )

    rng = np.random.default_rng(settings.random_seed)
    samples = rng.normal(loc=np.array(input_lab, dtype=float), scale=settings.sigma_vector, size=(settings.simulations, 3))
    samples[:, 0] = np.clip(samples[:, 0], 0.0, 100.0)
    samples[:, 1] = np.clip(samples[:, 1], -128.0, 127.0)
    samples[:, 2] = np.clip(samples[:, 2], -128.0, 127.0)

    matched_names: List[str] = []
    best_deltas: List[float] = []
    for sample in samples:
        delta_e_values = calculate_delta_e(
            sample.tolist(),
            dataset_df,
            method=method,
            delta_e_preset=delta_e_preset,
            science_settings=science_settings,
        )
        if delta_e_values.size == 0 or np.all(np.isnan(delta_e_values)):
            continue
        min_idx = int(np.nanargmin(delta_e_values))
        matched_names.append(str(dataset_df.iloc[min_idx]["Color Name"]))
        best_deltas.append(float(delta_e_values[min_idx]))

    simulations_run = len(matched_names)
    if simulations_run == 0:
        return MonteCarloStabilitySummary(
            enabled=True,
            simulations_requested=settings.simulations,
            simulations_run=0,
            top_match_repeat_rate=0.0,
            stability_score=0.0,
            stability_label="Low",
            mean_best_delta_e=None,
            std_best_delta_e=None,
            p05_best_delta_e=None,
            p95_best_delta_e=None,
            competitor_breakdown=[],
            is_unstable=True,
            message="No valid Monte Carlo draws produced a match.",
        )

    counts = pd.Series(matched_names).value_counts()
    baseline_count = int(counts.get(str(baseline_match_name), 0))
    repeat_rate = baseline_count / simulations_run
    stability_label = stability_label_from_repeat_rate(repeat_rate)
    stability_score = round(repeat_rate * 100.0, 1)

    best_delta_arr = np.array(best_deltas, dtype=float)
    mean_best_delta = float(np.mean(best_delta_arr))
    std_best_delta = float(np.std(best_delta_arr))
    p05_best_delta = float(np.quantile(best_delta_arr, 0.05))
    p95_best_delta = float(np.quantile(best_delta_arr, 0.95))

    competitor_breakdown = [
        {
            "Color Name": str(name),
            "Count": int(count),
            "Rate": float(count / simulations_run),
        }
        for name, count in counts.items()
    ]
    is_unstable = repeat_rate < UNCERTAINTY_REPEAT_RATE_MODERATE
    if is_unstable:
        message = (
            f"Unstable under uncertainty: baseline match repeats in {repeat_rate * 100:.1f}% of simulations."
        )
    elif repeat_rate < UNCERTAINTY_REPEAT_RATE_HIGH:
        message = (
            f"Moderate stability: baseline match repeats in {repeat_rate * 100:.1f}% of simulations."
        )
    else:
        message = (
            f"Stable under uncertainty: baseline match repeats in {repeat_rate * 100:.1f}% of simulations."
        )

    return MonteCarloStabilitySummary(
        enabled=True,
        simulations_requested=settings.simulations,
        simulations_run=simulations_run,
        top_match_repeat_rate=repeat_rate,
        stability_score=stability_score,
        stability_label=stability_label,
        mean_best_delta_e=mean_best_delta,
        std_best_delta_e=std_best_delta,
        p05_best_delta_e=p05_best_delta,
        p95_best_delta_e=p95_best_delta,
        competitor_breakdown=competitor_breakdown,
        is_unstable=is_unstable,
        message=message,
    )


def find_closest_color(
    input_lab: Union[List[float], Tuple[float, float, float]],
    dataset_df: pd.DataFrame,
    delta_e_method: str = "euclidean",
    delta_e_preset: str = "paint",
    science_settings: Optional[ColorScienceSettings] = None,
) -> Tuple[Optional[pd.Series], Optional[float]]:
    """
    Finds the closest color from the dataset to the input LAB color based on Delta-E.
    Returns the closest color row and its Delta-E value.
    """
    if dataset_df.empty:
        _log_and_report("Dataset is empty after validation.", "error", "Dataset Error")
        return None, None

    ranked = build_ranked_matches(
        dataset_df,
        input_lab,
        method=delta_e_method,
        delta_e_preset=delta_e_preset,
        science_settings=science_settings,
    )
    if ranked.empty:
        _log_and_report("No valid Delta-E values were produced for matching.", "error", "Processing Error")
        return None, None
    closest_color = ranked.iloc[0]
    min_delta_e = float(closest_color["Delta E"])
    return closest_color, min_delta_e


def axis_coverage_ratio(
    values: pd.Series,
    value_range: Tuple[float, float],
    bins: int = QA_COVERAGE_BINS,
) -> float:
    arr = values.dropna().astype(float).values
    if arr.size == 0:
        return 0.0
    edges = np.linspace(value_range[0], value_range[1], bins + 1)
    hist, _ = np.histogram(arr, bins=edges)
    occupied = int((hist > 0).sum())
    return occupied / bins


def detect_iqr_outliers(dataset_df: pd.DataFrame) -> pd.Series:
    if dataset_df.empty or len(dataset_df) < 4:
        return pd.Series(False, index=dataset_df.index)
    mask = pd.Series(False, index=dataset_df.index)
    for col in ["L", "A", "B"]:
        q1 = float(dataset_df[col].quantile(0.25))
        q3 = float(dataset_df[col].quantile(0.75))
        iqr = q3 - q1
        if iqr <= 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = mask | ((dataset_df[col] < lower) | (dataset_df[col] > upper))
    return mask


def build_dataset_quality_report(
    dataset_df: pd.DataFrame,
    rows_original: int,
    invalid_rows_removed: int,
    out_of_range_rows_removed: int,
    duplicate_rows_removed: int,
) -> DatasetQualityReport:
    outlier_mask = detect_iqr_outliers(dataset_df)
    outlier_rows = int(outlier_mask.sum())

    l_coverage = axis_coverage_ratio(dataset_df["L"], LAB_L_RANGE, QA_COVERAGE_BINS)
    a_coverage = axis_coverage_ratio(dataset_df["A"], LAB_A_RANGE, QA_COVERAGE_BINS)
    b_coverage = axis_coverage_ratio(dataset_df["B"], LAB_B_RANGE, QA_COVERAGE_BINS)
    sparse_axes: List[str] = []
    if l_coverage < QA_MIN_COVERAGE_RATIO:
        sparse_axes.append("L")
    if a_coverage < QA_MIN_COVERAGE_RATIO:
        sparse_axes.append("A")
    if b_coverage < QA_MIN_COVERAGE_RATIO:
        sparse_axes.append("B")

    return DatasetQualityReport(
        rows_original=rows_original,
        rows_final=len(dataset_df),
        invalid_rows_removed=invalid_rows_removed,
        out_of_range_rows_removed=out_of_range_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
        outlier_rows=outlier_rows,
        l_coverage_ratio=l_coverage,
        a_coverage_ratio=a_coverage,
        b_coverage_ratio=b_coverage,
        sparse_axes=tuple(sparse_axes),
    )


@st.cache_data(show_spinner=True)
def load_dataset(uploaded_file: IO[Any]) -> Tuple[pd.DataFrame, DatasetQualityReport]:
    """
    Loads and validates the CSV dataset containing LAB values and color names.
    Runs data quality checks: invalid values, range checks, duplicates, outliers, and coverage.
    """
    try:
        raw_df = read_uploaded_csv_flexible(uploaded_file)
    except DatasetError as exc:
        _log_and_report(str(exc), "error", "File Reading Error")
        raise DatasetError("Failed to load dataset.") from exc

    try:
        dataset_df, column_map = standardize_lab_dataframe(raw_df, default_name_prefix="Color")
    except DatasetError as exc:
        _log_and_report(str(exc), "error", "Dataset Error")
        raise DatasetError("Dataset missing required LAB fields.") from exc

    mapped_columns = {k: v for k, v in column_map.items() if v not in {k, "<generated>"}}
    if mapped_columns:
        mapping_summary = ", ".join(f"{dst} <- {src}" for dst, src in mapped_columns.items())
        _log_and_report(
            f"Auto-mapped dataset columns: {mapping_summary}",
            "info",
        )

    rows_original = len(dataset_df)
    dataset_df["Color Name"] = dataset_df["Color Name"].astype(str).str.strip()
    for col in ["L", "A", "B"]:
        dataset_df[col] = coerce_lab_series(dataset_df[col])

    invalid_rows_mask = dataset_df[list(REQUIRED_COLUMNS)].isnull().any(axis=1)
    invalid_rows_removed = int(invalid_rows_mask.sum())
    if invalid_rows_removed > 0:
        dataset_df = dataset_df.loc[~invalid_rows_mask].copy()

    out_of_range_mask = (
        (dataset_df["L"] < LAB_L_RANGE[0])
        | (dataset_df["L"] > LAB_L_RANGE[1])
        | (dataset_df["A"] < LAB_A_RANGE[0])
        | (dataset_df["A"] > LAB_A_RANGE[1])
        | (dataset_df["B"] < LAB_B_RANGE[0])
        | (dataset_df["B"] > LAB_B_RANGE[1])
    )
    out_of_range_rows_removed = int(out_of_range_mask.sum())
    if out_of_range_rows_removed > 0:
        dataset_df = dataset_df.loc[~out_of_range_mask].copy()

    duplicate_rows_mask = dataset_df.duplicated(subset=["L", "A", "B", "Color Name"], keep="first")
    duplicate_rows_removed = int(duplicate_rows_mask.sum())
    if duplicate_rows_removed > 0:
        dataset_df = dataset_df.loc[~duplicate_rows_mask].copy()

    dataset_df = dataset_df.reset_index(drop=True)
    if dataset_df.empty:
        _log_and_report(
            "CSV contains no valid rows after QA cleaning. Check missing, range, and duplicate issues.",
            "error",
            "Dataset Error",
        )
        raise DatasetError("Dataset is empty.")

    quality_report = build_dataset_quality_report(
        dataset_df,
        rows_original=rows_original,
        invalid_rows_removed=invalid_rows_removed,
        out_of_range_rows_removed=out_of_range_rows_removed,
        duplicate_rows_removed=duplicate_rows_removed,
    )
    return dataset_df, quality_report


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


@st.cache_data(show_spinner=True)
def extract_alternative_terms_json(json_file: IO[Any]) -> pd.DataFrame:
    """
    Extracts alternative color terms from a Getty AAT JSON document.
    Pulls terms from identified_by (preferred/alternate labels) and subject_of descriptions.
    """
    try:
        data = json.load(json_file)
    except Exception as exc:
        _log_and_report(f"Failed to parse JSON file: {exc}", "error", "JSON Parsing Error")
        raise RDFParsingError("Failed to parse JSON file.")

    rows: List[Dict[str, str]] = []

    def get_lang(obj: Any) -> str:
        if isinstance(obj, dict):
            lang = obj.get("language")
            if isinstance(lang, list) and lang:
                return lang[0].get("_label") or lang[0].get("id") or "unknown"
            if isinstance(lang, dict):
                return lang.get("_label") or lang.get("id") or "unknown"
        return "unknown"

    def add_term(term: Any, lang: str, skip_digit_filter: bool = False) -> None:
        if not isinstance(term, str):
            return
        term_text = term.strip()
        if not term_text:
            return
        if not skip_digit_filter:
            if "centroid" in term_text.lower():
                rows.append({"Term": term_text, "Language": lang or "unknown"})
                return
            if re.search(r"\d", term_text) or len(term_text) > 50:
                return
        else:
            if len(term_text) > 100:
                return
        rows.append({"Term": term_text, "Language": lang or "unknown"})

    def process_record(record: Dict[str, Any]) -> None:
        if not isinstance(record, dict):
            return
        add_term(record.get("_label"), "unknown", skip_digit_filter=False)

        identified = record.get("identified_by", [])
        if isinstance(identified, dict):
            identified = [identified]
        if isinstance(identified, list):
            for name_obj in identified:
                if not isinstance(name_obj, dict):
                    continue
                lang = get_lang(name_obj)
                add_term(name_obj.get("content"), lang)
                alternatives = name_obj.get("alternative", [])
                if isinstance(alternatives, dict):
                    alternatives = [alternatives]
                if isinstance(alternatives, list):
                    for alt in alternatives:
                        if not isinstance(alt, dict):
                            continue
                        alt_lang = get_lang(alt) or lang
                        add_term(alt.get("content"), alt_lang)

        subject_of = record.get("subject_of", [])
        if isinstance(subject_of, dict):
            subject_of = [subject_of]
        if isinstance(subject_of, list):
            for desc in subject_of:
                if not isinstance(desc, dict):
                    continue
                lang = get_lang(desc)
                add_term(desc.get("content"), lang, skip_digit_filter=True)

    if isinstance(data, dict):
        process_record(data)
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                process_record(item)

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
def cached_lab_to_rgb(
    l: float,
    a: float,
    b: float,
    source_illuminant: str = "d50",
    target_illuminant: str = "d65",
    observer: str = "2",
    chromatic_adaptation: str = "bradford",
) -> Tuple[int, int, int]:
    """Cached helper for converting LAB to RGB."""
    return lab_to_rgb(
        [l, a, b],
        source_illuminant=source_illuminant,
        target_illuminant=target_illuminant,
        observer=observer,
        chromatic_adaptation=chromatic_adaptation,
    )


def create_pairwise_scatter_matrix(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    closest_lab: List[float],
    science_settings: ColorScienceSettings,
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
        lambda row: format_rgb(
            cached_lab_to_rgb(
                row["L"],
                row["A"],
                row["B"],
                source_illuminant=science_settings.source_illuminant,
                target_illuminant=science_settings.target_illuminant,
                observer=science_settings.observer,
                chromatic_adaptation=science_settings.chromatic_adaptation,
            )
        ),
        axis=1,
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

def render_section_header(title: str, subtitle: Optional[str] = None, accent: bool = False) -> None:
    subtitle_html = f"<div class='section-subtitle'>{subtitle}</div>" if subtitle else ""
    accent_html = "<div class='section-accent'></div>" if accent else ""
    st.markdown(
        f"""
        <div class="section-heading">
            <div class="section-title">{title}</div>
            {subtitle_html}
            {accent_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero(
    input_rgb: Tuple[int, int, int],
    input_lab: List[float],
    method_label: str,
    science_settings: ColorScienceSettings,
    input_gamut: Optional[RgbGamutDiagnostics] = None,
) -> None:
    swatch_color = format_rgb(input_rgb)
    gamut_line = ""
    if input_gamut is not None:
        if input_gamut.out_of_gamut:
            gamut_line = f"<div>Gamut: Out of sRGB ({input_gamut.channels_label})</div>"
        else:
            gamut_line = "<div>Gamut: In sRGB</div>"
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
                    <div>Whitepoint: {science_settings.source_label} to {science_settings.target_label} ({science_settings.observer_label})</div>
                    <div>Adaptation: {science_settings.adaptation_label}</div>
                    <div>LAB: L {input_lab[0]:.1f} | A {input_lab[1]:.1f} | B {input_lab[2]:.1f}</div>
                    <div>RGB: {rgb_to_hex(input_rgb)}</div>
                    {gamut_line}
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
    st.markdown(
        f"<div class='status-wrap'><div class='status-pill'>{text}</div></div>",
        unsafe_allow_html=True,
    )


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
    rows: List[Dict[str, str]] = [
        {
            "Metric": "Input LAB",
            "Value": f"L={result.input_lab[0]:.2f}, A={result.input_lab[1]:.2f}, B={result.input_lab[2]:.2f}",
        },
        {"Metric": "Closest ISCC-NBS Color", "Value": result.closest_name},
        {"Metric": f"Delta-E ({result.method_label})", "Value": result.delta_e_label},
        {"Metric": "Delta-E Domain Preset", "Value": DELTA_E_PRESET_LABELS[normalize_delta_e_preset(result.delta_e_preset)]},
        {"Metric": "Match Confidence", "Value": result.confidence.score_label},
        {
            "Metric": "Top-2 Delta-E Margin",
            "Value": f"{result.confidence.margin_top2:.2f}" if result.confidence.margin_top2 is not None else "N/A",
        },
        {
            "Metric": "Top-3 Delta-E Margin",
            "Value": f"{result.confidence.margin_top3:.2f}" if result.confidence.margin_top3 is not None else "N/A",
        },
        {
            "Metric": "Ambiguity Flag",
            "Value": "Yes" if result.confidence.is_ambiguous else "No",
        },
        {
            "Metric": "Closest LAB",
            "Value": f"L={result.closest_lab[0]:.2f}, A={result.closest_lab[1]:.2f}, B={result.closest_lab[2]:.2f}",
        },
        {"Metric": "Input RGB", "Value": rgb_to_hex(result.input_rgb)},
        {"Metric": "Input Gamut Status", "Value": result.input_gamut.status_label},
        {"Metric": "Input Clipped Channels", "Value": result.input_gamut.channels_label},
        {"Metric": "Closest RGB", "Value": rgb_to_hex(result.closest_rgb)},
        {"Metric": "Closest Gamut Status", "Value": result.closest_gamut.status_label},
        {"Metric": "Closest Clipped Channels", "Value": result.closest_gamut.channels_label},
        {
            "Metric": "LAB Whitepoint",
            "Value": f"{result.science_settings.source_label} ({result.science_settings.observer_label})",
        },
        {"Metric": "RGB Whitepoint", "Value": result.science_settings.target_label},
        {"Metric": "Chromatic Adaptation", "Value": result.science_settings.adaptation_label},
    ]

    if result.uncertainty.enabled:
        rows.extend(
            [
                {"Metric": "Monte Carlo Draws", "Value": str(result.uncertainty.simulations_run)},
                {"Metric": "Top Match Repeat Rate", "Value": result.uncertainty.repeat_rate_label},
                {"Metric": "Uncertainty Stability", "Value": result.uncertainty.stability_score_label},
                {"Metric": "MC Best Delta-E Mean +/- SD", "Value": result.uncertainty.delta_band_label},
                {"Metric": "MC Best Delta-E p05-p95", "Value": result.uncertainty.percentile_band_label},
            ]
        )
    else:
        rows.append({"Metric": "Monte Carlo Stability", "Value": "Disabled"})

    return pd.DataFrame(rows)

# =============================================================================
# ColorAnalyzer Class
# =============================================================================

class ColorAnalyzer:
    """Encapsulates dataset loading, RDF processing, matching, and visualization."""

    REQUIRED_COLUMNS = REQUIRED_COLUMNS

    def __init__(self) -> None:
        self.dataset_df: Optional[pd.DataFrame] = None
        self.dataset_quality_report: Optional[DatasetQualityReport] = None
        self.rdf_alternatives_df: Optional[pd.DataFrame] = None
        self.input_lab: Optional[List[float]] = None
        self.delta_e_method: str = "euclidean"
        self.delta_e_preset: str = "paint"
        self.science_settings: ColorScienceSettings = ColorScienceSettings()
        self.uncertainty_settings: MeasurementUncertaintySettings = MeasurementUncertaintySettings()
        self.result: Optional[ColorMatchResult] = None

    def load_dataset(self, uploaded_file: IO[Any]) -> bool:
        """Loads and validates the dataset CSV file."""
        try:
            self.dataset_df, self.dataset_quality_report = load_dataset(uploaded_file)
        except DatasetError:
            return False
        return True

    def load_rdf(self, rdf_file: Optional[IO[Any]]) -> None:
        """Processes the RDF file and stores alternative color terms."""
        if rdf_file is not None:
            file_name = getattr(rdf_file, "name", "").lower()
            try:
                data_bytes = rdf_file.getvalue() if hasattr(rdf_file, "getvalue") else rdf_file.read()
            except Exception:
                data_bytes = None

            if data_bytes is None:
                try:
                    self.rdf_alternatives_df = extract_alternative_terms_rdf(rdf_file)
                except RDFParsingError:
                    self.rdf_alternatives_df = None
                return

            is_json = file_name.endswith(".json") or data_bytes.lstrip().startswith((b"{", b"["))
            if is_json:
                try:
                    self.rdf_alternatives_df = extract_alternative_terms_json(io.BytesIO(data_bytes))
                except RDFParsingError:
                    self.rdf_alternatives_df = None
            else:
                try:
                    self.rdf_alternatives_df = extract_alternative_terms_rdf(io.BytesIO(data_bytes))
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

    def set_delta_e_preset(self, preset: str) -> None:
        """Sets method parameter presets for CIE94 and CMC."""
        self.delta_e_preset = normalize_delta_e_preset(preset)

    def set_color_science_settings(self, settings: ColorScienceSettings) -> None:
        """Sets illuminant/observer/adaptation settings."""
        self.science_settings = settings.normalized()

    def set_uncertainty_settings(self, settings: MeasurementUncertaintySettings) -> None:
        """Sets measurement uncertainty and Monte Carlo simulation settings."""
        self.uncertainty_settings = settings.normalized()

    def match_color(self) -> Optional[ColorMatchResult]:
        """Finds the closest dataset color to the input LAB color."""
        if self.dataset_df is None or self.input_lab is None:
            _log_and_report("Dataset or input LAB color not set.", "error", "Processing Error")
            return None
        closest_color, delta_e = find_closest_color(
            self.input_lab,
            self.dataset_df,
            delta_e_method=self.delta_e_method,
            delta_e_preset=self.delta_e_preset,
            science_settings=self.science_settings,
        )
        if closest_color is None or delta_e is None:
            return None
        confidence = compute_match_confidence(
            self.dataset_df,
            self.input_lab,
            method=self.delta_e_method,
            delta_e_preset=self.delta_e_preset,
            science_settings=self.science_settings,
        )
        uncertainty = compute_uncertainty_stability(
            self.dataset_df,
            self.input_lab,
            baseline_match_name=str(closest_color["Color Name"]),
            method=self.delta_e_method,
            delta_e_preset=self.delta_e_preset,
            science_settings=self.science_settings,
            uncertainty_settings=self.uncertainty_settings,
        )
        closest_lab = [closest_color["L"], closest_color["A"], closest_color["B"]]
        input_rgb, input_gamut = lab_to_rgb_with_diagnostics(
            self.input_lab,
            source_illuminant=self.science_settings.source_illuminant,
            target_illuminant=self.science_settings.target_illuminant,
            observer=self.science_settings.observer,
            chromatic_adaptation=self.science_settings.chromatic_adaptation,
        )
        closest_rgb, closest_gamut = lab_to_rgb_with_diagnostics(
            closest_lab,
            source_illuminant=self.science_settings.source_illuminant,
            target_illuminant=self.science_settings.target_illuminant,
            observer=self.science_settings.observer,
            chromatic_adaptation=self.science_settings.chromatic_adaptation,
        )
        result = ColorMatchResult(
            input_lab=self.input_lab,
            closest_lab=closest_lab,
            closest_name=closest_color["Color Name"],
            delta_e=float(delta_e),
            input_rgb=input_rgb,
            closest_rgb=closest_rgb,
            input_gamut=input_gamut,
            closest_gamut=closest_gamut,
            method=self.delta_e_method,
            delta_e_preset=self.delta_e_preset,
            confidence=confidence,
            uncertainty_settings=self.uncertainty_settings,
            uncertainty=uncertainty,
            science_settings=self.science_settings,
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
                calculate_delta_e(
                    result.input_lab,
                    self.dataset_df,
                    method=self.delta_e_method,
                    delta_e_preset=self.delta_e_preset,
                    science_settings=self.science_settings,
                )
            ),
            "density_heatmap": create_color_density_heatmap(self.dataset_df),
            "scatter_matrix": create_pairwise_scatter_matrix(
                self.dataset_df,
                result.input_lab,
                result.closest_lab,
                self.science_settings,
            ),
        }
        return figs

# =============================================================================
# Streamlit UI Functions
# =============================================================================

def display_sidebar() -> Tuple[Any, Any, List[float], str, str, ColorScienceSettings, MeasurementUncertaintySettings]:
    """Displays the sidebar widgets for file uploads and LAB input."""
    st.sidebar.header("Input Studio")
    st.sidebar.markdown("Upload your datasets and tune the LAB values.")
    st.sidebar.markdown("### Device Profile")
    device_profile = st.sidebar.selectbox(
        "Capture workflow",
        ("General LAB", NIX_MINI3_PROFILE_LABEL),
        index=0,
        help="Use the Nix profile for direct Nix scan import and multipoint aggregation.",
    )
    is_nix_profile = device_profile == NIX_MINI3_PROFILE_LABEL
    if is_nix_profile:
        st.sidebar.caption(
            "Nix profile: optimized for imported Nix scans, CIEDE2000 matching, and repeatability-aware input."
        )
    st.sidebar.markdown("### Uploads")
    csv_file = st.sidebar.file_uploader("ISCC-NBS LAB CSV", type=["csv"])
    rdf_file = st.sidebar.file_uploader("Getty AAT RDF / JSON", type=["xml", "json"])
    st.sidebar.markdown(
        """
        <div class="sidebar-card">
            <h4>Workflow</h4>
            <ol>
                <li>Upload the ISCC-NBS LAB dataset.</li>
                <li>Optionally upload the Getty AAT RDF or JSON file.</li>
                <li>Choose an input mode (LAB, RGB/HEX, picker, image, or Nix scan import).</li>
                <li>Pick a Delta-E method.</li>
                <li>Analyze the closest color match.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown('<div class="sidebar-divider"></div>', unsafe_allow_html=True)
    st.sidebar.markdown("### Controls")
    delta_e_metric = st.sidebar.selectbox(
        "Delta-E method",
        DELTA_E_METHOD_OPTIONS,
        index=DELTA_E_METHOD_OPTIONS.index("CIEDE2000"),
    )
    delta_e_preset_label = st.sidebar.radio(
        "Domain preset",
        DELTA_E_PRESET_OPTIONS,
        horizontal=True,
        index=DELTA_E_PRESET_OPTIONS.index("Paint"),
        help="Applies CIE94/CMC parameters for paint or textile workflows.",
    )
    if is_nix_profile and delta_e_metric != "CIEDE2000":
        st.sidebar.caption("Tip for Nix workflows: CIEDE2000 is usually the most perceptually robust default.")
    delta_e_preset = DELTA_E_PRESET_UI_MAP[delta_e_preset_label]
    if delta_e_metric == "CIE94":
        cie94_profile = DELTA_E_PRESET_PROFILES[delta_e_preset]["cie94"]
        st.sidebar.caption(
            f"CIE94 params: K_L={cie94_profile['K_L']}, K_1={cie94_profile['K_1']}, K_2={cie94_profile['K_2']}."
        )
    elif delta_e_metric == "CMC l:c":
        cmc_profile = DELTA_E_PRESET_PROFILES[delta_e_preset]["cmc"]
        st.sidebar.caption(
            f"CMC params: l:c = {int(cmc_profile['pl'])}:{int(cmc_profile['pc'])}."
        )
    st.sidebar.markdown("### Color Science")
    source_illuminant = st.sidebar.selectbox(
        "LAB illuminant",
        [ill.upper() for ill in ILLUMINANT_OPTIONS],
        index=ILLUMINANT_OPTIONS.index("d50"),
        help="Whitepoint used to interpret input and dataset LAB coordinates.",
    )
    observer = st.sidebar.radio(
        "Observer",
        ("2", "10"),
        horizontal=True,
        index=0,
        help="Standard observer angle for tristimulus values.",
    )
    target_illuminant = st.sidebar.selectbox(
        "RGB target illuminant",
        [ill.upper() for ill in ILLUMINANT_OPTIONS],
        index=ILLUMINANT_OPTIONS.index("d65"),
        help="Target whitepoint for LAB->RGB preview rendering.",
    )
    adaptation_label = st.sidebar.selectbox(
        "Chromatic adaptation",
        ("Bradford", "Von Kries", "XYZ Scaling", "None"),
        index=0,
        help="Adapt colors between LAB and RGB whitepoints.",
    )
    science_settings = ColorScienceSettings(
        source_illuminant=source_illuminant.lower(),
        target_illuminant=target_illuminant.lower(),
        observer=observer,
        chromatic_adaptation=CHROMATIC_ADAPTATION_UI_MAP[adaptation_label],
    ).normalized()
    if science_settings.chromatic_adaptation == "none" and science_settings.source_illuminant != science_settings.target_illuminant:
        st.sidebar.caption(
            "Chromatic adaptation is disabled, so matching and RGB preview stay in the LAB source whitepoint."
        )
    st.sidebar.markdown("### Measurement Uncertainty")
    default_sigma = 0.35 if is_nix_profile else UNCERTAINTY_DEFAULT_SIGMA
    default_simulations = 500 if is_nix_profile else UNCERTAINTY_DEFAULT_SIMULATIONS
    enable_uncertainty = st.sidebar.checkbox(
        "Run Monte Carlo stability",
        value=is_nix_profile,
        help="Simulate measurement noise to estimate how often the top match remains stable.",
    )
    l_sigma = default_sigma
    a_sigma = default_sigma
    b_sigma = default_sigma
    simulations = default_simulations
    random_seed = 42
    if enable_uncertainty:
        l_sigma = st.sidebar.slider(
            "L sigma",
            UNCERTAINTY_MIN_SIGMA,
            UNCERTAINTY_MAX_SIGMA,
            default_sigma,
            0.1,
            help="Expected standard deviation for L measurements.",
        )
        a_sigma = st.sidebar.slider(
            "A sigma",
            UNCERTAINTY_MIN_SIGMA,
            UNCERTAINTY_MAX_SIGMA,
            default_sigma,
            0.1,
            help="Expected standard deviation for A measurements.",
        )
        b_sigma = st.sidebar.slider(
            "B sigma",
            UNCERTAINTY_MIN_SIGMA,
            UNCERTAINTY_MAX_SIGMA,
            default_sigma,
            0.1,
            help="Expected standard deviation for B measurements.",
        )
        simulations = st.sidebar.slider(
            "Monte Carlo draws",
            UNCERTAINTY_MIN_SIMULATIONS,
            UNCERTAINTY_MAX_SIMULATIONS,
            default_simulations,
            50,
            help="More draws improve estimate quality but increase runtime.",
        )
        random_seed = int(
            st.sidebar.number_input(
                "Random seed",
                min_value=0,
                max_value=1000000,
                value=42,
                step=1,
            )
        )
        st.sidebar.caption("Uncertainty simulation can noticeably increase analysis time.")

    uncertainty_settings = MeasurementUncertaintySettings(
        enabled=enable_uncertainty,
        l_sigma=l_sigma,
        a_sigma=a_sigma,
        b_sigma=b_sigma,
        simulations=simulations,
        random_seed=random_seed,
    ).normalized()

    st.sidebar.markdown("### Color Input")
    input_mode_options = (
        "LAB Slider",
        "LAB Manual",
        "RGB / HEX",
        "Color Picker",
        "Image Sampler",
        f"{NIX_MINI3_PROFILE_LABEL} Capture",
    )
    default_input_mode = f"{NIX_MINI3_PROFILE_LABEL} Capture" if is_nix_profile else "LAB Slider"
    input_mode = st.sidebar.radio(
        "Input mode",
        input_mode_options,
        index=input_mode_options.index(default_input_mode),
    )

    if input_mode == "LAB Slider":
        lab_l = st.sidebar.slider("L", 0.0, 100.0, 50.0, 0.1)
        lab_a = st.sidebar.slider("A", -128.0, 127.0, 0.0, 0.1)
        lab_b = st.sidebar.slider("B", -128.0, 127.0, 0.0, 0.1)
        input_lab = [lab_l, lab_a, lab_b]
    elif input_mode == "LAB Manual":
        lab_l = st.sidebar.number_input(
            "L (0-100)",
            min_value=0.0,
            max_value=100.0,
            value=50.0,
            step=0.1,
        )
        lab_a = st.sidebar.number_input(
            "A (-128 to 127)",
            min_value=-128.0,
            max_value=127.0,
            value=0.0,
            step=0.1,
        )
        lab_b = st.sidebar.number_input(
            "B (-128 to 127)",
            min_value=-128.0,
            max_value=127.0,
            value=0.0,
            step=0.1,
        )
        input_lab = [lab_l, lab_a, lab_b]
    elif input_mode == "RGB / HEX":
        st.sidebar.caption("Enter a HEX value or RGB channels. HEX takes priority when valid.")
        hex_input = st.sidebar.text_input("HEX (#RRGGBB or #RGB)", value="#808080")
        rgb_cols = st.sidebar.columns(3)
        rgb_r = int(rgb_cols[0].number_input("R", min_value=0, max_value=255, value=128, step=1))
        rgb_g = int(rgb_cols[1].number_input("G", min_value=0, max_value=255, value=128, step=1))
        rgb_b = int(rgb_cols[2].number_input("B", min_value=0, max_value=255, value=128, step=1))
        parsed_hex = parse_hex_color(hex_input)
        if parsed_hex is not None:
            active_rgb = parsed_hex
            st.sidebar.caption(f"Using HEX RGB: {active_rgb[0]}, {active_rgb[1]}, {active_rgb[2]}")
        else:
            active_rgb = (rgb_r, rgb_g, rgb_b)
            if hex_input.strip():
                st.sidebar.caption("HEX is invalid; using RGB channels.")
        input_lab = rgb_to_lab(
            active_rgb,
            source_illuminant=science_settings.source_illuminant,
            observer=science_settings.observer,
        )
        st.sidebar.caption(f"Converted LAB: L {input_lab[0]:.2f}, A {input_lab[1]:.2f}, B {input_lab[2]:.2f}")
    elif input_mode == "Color Picker":
        picked_hex = st.sidebar.color_picker("Pick an sRGB color", value="#808080")
        picked_rgb = parse_hex_color(picked_hex) or (128, 128, 128)
        st.sidebar.caption(f"Picked RGB: {picked_rgb[0]}, {picked_rgb[1]}, {picked_rgb[2]}")
        input_lab = rgb_to_lab(
            picked_rgb,
            source_illuminant=science_settings.source_illuminant,
            observer=science_settings.observer,
        )
        st.sidebar.caption(f"Converted LAB: L {input_lab[0]:.2f}, A {input_lab[1]:.2f}, B {input_lab[2]:.2f}")
    elif input_mode == "Image Sampler":
        if Image is None:
            st.sidebar.error("Image sampling requires Pillow; using neutral LAB fallback.")
            input_lab = [50.0, 0.0, 0.0]
        else:
            sample_image_file = st.sidebar.file_uploader(
                "Upload image for pixel sampling",
                type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
                key="pixel_sampler_image",
            )
            if sample_image_file is None:
                st.sidebar.info("Upload an image to sample a pixel color.")
                input_lab = [50.0, 0.0, 0.0]
            else:
                try:
                    with Image.open(sample_image_file) as image_obj:
                        sampled_image = image_obj.convert("RGB")
                        width, height = sampled_image.size
                        st.sidebar.image(
                            sampled_image,
                            caption=f"Sample image ({width}x{height})",
                            use_container_width=True,
                        )
                        pixel_x = st.sidebar.slider("Pixel X", 0, width - 1, width // 2)
                        pixel_y = st.sidebar.slider("Pixel Y", 0, height - 1, height // 2)
                        pixel_rgb_raw = sampled_image.getpixel((pixel_x, pixel_y))
                        if isinstance(pixel_rgb_raw, int):
                            sampled_rgb = (pixel_rgb_raw, pixel_rgb_raw, pixel_rgb_raw)
                        else:
                            sampled_rgb = (int(pixel_rgb_raw[0]), int(pixel_rgb_raw[1]), int(pixel_rgb_raw[2]))
                except Exception as exc:
                    st.sidebar.error(f"Unable to sample the uploaded image: {exc}")
                    input_lab = [50.0, 0.0, 0.0]
                else:
                    st.sidebar.caption(
                        f"Sampled RGB: {sampled_rgb[0]}, {sampled_rgb[1]}, {sampled_rgb[2]} at ({pixel_x}, {pixel_y})"
                    )
                    st.sidebar.markdown(
                        f"<div style='height:24px;border-radius:999px;border:1px solid rgba(0,0,0,0.15);background:{rgb_to_hex(sampled_rgb)};'></div>",
                        unsafe_allow_html=True,
                    )
                    input_lab = rgb_to_lab(
                        sampled_rgb,
                        source_illuminant=science_settings.source_illuminant,
                        observer=science_settings.observer,
                    )
                    st.sidebar.caption(
                        f"Converted LAB: L {input_lab[0]:.2f}, A {input_lab[1]:.2f}, B {input_lab[2]:.2f}"
                    )
    else:
        st.sidebar.caption("Import one or many Nix scans and aggregate to a single LAB target.")
        source_mode = st.sidebar.radio(
            "Nix scan source",
            ("Upload Nix CSV", "Paste LAB values"),
            horizontal=True,
        )
        aggregation_label = st.sidebar.selectbox(
            "Scan aggregation",
            ("Median (recommended)", "Mean", "Latest"),
            index=0,
            help="Median is robust to occasional bad scans; mean is smoother; latest uses the most recent scan.",
        )
        aggregation_key = {
            "Median (recommended)": "median",
            "Mean": "mean",
            "Latest": "latest",
        }[aggregation_label]
        measurements_df = pd.DataFrame(columns=["Color Name", "L", "A", "B"])
        dropped_rows = 0

        if source_mode == "Upload Nix CSV":
            nix_scan_file = st.sidebar.file_uploader(
                "Nix measurement CSV",
                type=["csv"],
                key="nix_measurement_csv",
            )
            if nix_scan_file is not None:
                try:
                    measurements_df, column_map, dropped_rows = load_lab_measurements_csv(
                        nix_scan_file,
                        default_name_prefix="Scan",
                    )
                except DatasetError as exc:
                    st.sidebar.error(f"Nix CSV parse error: {exc}")
                    measurements_df = pd.DataFrame(columns=["Color Name", "L", "A", "B"])
                else:
                    mapped_columns = {k: v for k, v in column_map.items() if v not in {k, "<generated>"}}
                    if mapped_columns:
                        st.sidebar.caption(
                            "Mapped columns: "
                            + ", ".join(f"{dst} <- {src}" for dst, src in mapped_columns.items())
                        )
        else:
            pasted_rows = st.sidebar.text_area(
                "Paste LAB scans",
                height=120,
                placeholder="L* 52.3 a* 14.1 b* 22.0\nL* 52.8 a* 13.8 b* 22.4\n52.1, 14.0, 21.9",
                help="One measurement per line. Supported formats include labeled LAB or comma/space-separated triplets.",
            )
            if pasted_rows.strip():
                parsed_df = parse_pasted_lab_measurements(pasted_rows)
                measurements_df, dropped_rows = clean_lab_measurements(parsed_df)

        if dropped_rows > 0:
            st.sidebar.caption(f"Dropped {dropped_rows} invalid or out-of-range scan row(s).")

        if measurements_df.empty:
            st.sidebar.info("Add Nix measurements to build the LAB input.")
            input_lab = [50.0, 0.0, 0.0]
        else:
            input_lab = aggregate_lab_measurements(measurements_df, aggregation=aggregation_key)
            spread = measurements_df[["L", "A", "B"]].std(ddof=0).fillna(0.0)
            st.sidebar.caption(
                f"Scans: {len(measurements_df)} | Sigma L/A/B: {spread['L']:.2f} / {spread['A']:.2f} / {spread['B']:.2f}"
            )
            st.sidebar.caption(
                f"Aggregated LAB: L {input_lab[0]:.2f}, A {input_lab[1]:.2f}, B {input_lab[2]:.2f}"
            )
            st.sidebar.dataframe(
                measurements_df.head(12),
                use_container_width=True,
                height=170,
            )
    method_key = DELTA_E_METHOD_UI_MAP[delta_e_metric]
    return csv_file, rdf_file, input_lab, method_key, delta_e_preset, science_settings, uncertainty_settings


def sparkline_sample(series: pd.Series, points: int = SPARKLINE_POINTS) -> List[float]:
    values = series.dropna().astype(float).values
    if values.size == 0:
        return []
    if values.size <= points:
        return values.tolist()
    quantiles = np.linspace(0, 1, points)
    return np.quantile(values, quantiles).tolist()


def infer_color_tags(row: pd.Series) -> List[str]:
    l_val = float(row["L"])
    a_val = float(row["A"])
    b_val = float(row["B"])
    chroma = (a_val**2 + b_val**2) ** 0.5
    tags: List[str] = []

    if l_val >= 70:
        tags.append("Light")
    elif l_val <= 35:
        tags.append("Deep")
    else:
        tags.append("Mid")

    if a_val >= 12 and b_val >= 12:
        tags.append("Warm")
    elif a_val <= -12 and b_val <= -12:
        tags.append("Cool")
    else:
        tags.append("Balanced")

    if chroma >= 60:
        tags.append("Vivid")
    elif chroma <= 25:
        tags.append("Muted")
    else:
        tags.append("Soft")

    return tags


def build_top_matches(
    dataset_df: pd.DataFrame,
    input_lab: List[float],
    method: str,
    delta_e_preset: str,
    science_settings: ColorScienceSettings,
    top_n: int = 10,
) -> pd.DataFrame:
    ranked = build_ranked_matches(
        dataset_df,
        input_lab,
        method,
        delta_e_preset,
        science_settings,
    )
    return ranked.head(top_n)[["Color Name", "Delta E"]]


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


def render_dataset_overview(
    dataset_df: pd.DataFrame,
    quality_report: Optional[DatasetQualityReport] = None,
) -> None:
    summary = summarize_dataset(dataset_df)
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)
    render_section_header("Dataset Overview", "A quick read on the ISCC-NBS archive.", accent=True)
    spark_l = sparkline_sample(dataset_df["L"])
    spark_a = sparkline_sample(dataset_df["A"])
    spark_b = sparkline_sample(dataset_df["B"])

    cols = st.columns([1.3, 1])
    cols[0].metric("Colors", f"{summary['rows']:,}", border=True)
    cols[1].metric("Unique Names", f"{summary['unique']:,}", border=True)

    cols = st.columns(3)
    cols[0].metric(
        "L Spread",
        f"{summary['l_min']:.1f} - {summary['l_max']:.1f}",
        chart_data=spark_l,
        border=True,
    )
    cols[1].metric(
        "A Spread",
        f"{summary['a_min']:.1f} - {summary['a_max']:.1f}",
        chart_data=spark_a,
        border=True,
    )
    cols[2].metric(
        "B Spread",
        f"{summary['b_min']:.1f} - {summary['b_max']:.1f}",
        chart_data=spark_b,
        border=True,
    )

    cols = st.columns(3)
    if quality_report is None:
        cols[0].metric("Dataset Health", "Validated", border=True)
        cols[1].metric("Color Space", "CIELAB", border=True)
        cols[2].metric("Ready", "Yes", border=True)
    else:
        cols[0].metric("Dataset Health", quality_report.health_label, border=True)
        cols[1].metric("Rows Removed (QA)", f"{quality_report.rows_removed_total}", border=True)
        cols[2].metric("Outlier Flags", f"{quality_report.outlier_rows}", border=True)

        coverage_cols = st.columns(2)
        coverage_cols[0].metric("Coverage (L/A/B)", quality_report.coverage_summary_label, border=True)
        coverage_cols[1].metric(
            "Coverage Gaps",
            ", ".join(quality_report.sparse_axes) if quality_report.sparse_axes else "None",
            border=True,
        )

        warnings = quality_report.warnings
        if warnings:
            st.warning("Dataset QA findings:\n- " + "\n- ".join(warnings))
        else:
            st.success("Dataset QA checks passed with no major warnings.")

    with st.expander("Preview dataset", expanded=False):
        preview_df = dataset_df.head(12).copy()
        preview_df["Color Tags"] = preview_df.apply(infer_color_tags, axis=1)
        st.dataframe(
            preview_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Color Tags": st.column_config.MultiselectColumn(
                    "Color Tags",
                    options=TAG_OPTIONS,
                    color=TAG_COLOR_PALETTE,
                    help="Auto-inferred tags derived from LAB values.",
                )
            },
        )


def render_input_preview(
    input_lab: List[float],
    input_rgb: Tuple[int, int, int],
    method_label: str,
    science_settings: ColorScienceSettings,
    input_gamut: Optional[RgbGamutDiagnostics] = None,
) -> None:
    render_section_header("Input Preview", "Dial in the LAB values before analysis.")
    cols = st.columns([1.2, 1])
    with cols[0]:
        render_color_card("Input LAB", method_label, input_rgb, input_lab)
    with cols[1]:
        st.markdown(
            f"""
            <div class="glass-card">
                <h3>Analysis Notes</h3>
                <p>Stay within valid LAB ranges for reliable conversion. Distance model: {method_label}.</p>
                <p>Observer {science_settings.observer_label}; LAB {science_settings.source_label}; RGB {science_settings.target_label}; Adaptation {science_settings.adaptation_label}.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    if input_gamut is not None:
        if input_gamut.out_of_gamut:
            st.warning(f"Input preview warning: {input_gamut.warning_message}")
    st.markdown('<div class="section-spacer-lg"></div>', unsafe_allow_html=True)


def render_results_section(
    result: ColorMatchResult,
    rdf_df: Optional[pd.DataFrame],
    dataset_df: pd.DataFrame,
) -> None:
    render_section_header("Match Studio", "Your closest ISCC-NBS match and key metrics.")
    cols = st.columns(5)
    cols[0].metric("Closest Match", result.closest_name)
    cols[1].metric("Delta-E", result.delta_e_label)
    cols[2].metric("Method", result.method_label)
    cols[3].metric("Observer / WP", f"{result.science_settings.observer_label}, {result.science_settings.source_label}")
    cols[4].metric("Confidence", result.confidence.score_label)

    confidence_cols = st.columns(3)
    confidence_cols[0].metric(
        "Top-2 Margin",
        f"{result.confidence.margin_top2:.2f}" if result.confidence.margin_top2 is not None else "N/A",
    )
    confidence_cols[1].metric(
        "Top-3 Margin",
        f"{result.confidence.margin_top3:.2f}" if result.confidence.margin_top3 is not None else "N/A",
    )
    confidence_cols[2].metric("Ambiguity", "Yes" if result.confidence.is_ambiguous else "No")

    if result.confidence.is_ambiguous:
        competing = ", ".join(result.confidence.competing_names[:3])
        details = f" Leading candidates: {competing}." if competing else ""
        st.warning(f"Ambiguous match. {result.confidence.ambiguity_reason}{details}")
    else:
        st.info(result.confidence.ambiguity_reason)

    gamut_warnings: List[str] = []
    if result.input_gamut.out_of_gamut:
        gamut_warnings.append(f"Input color: {result.input_gamut.warning_message}")
    if result.closest_gamut.out_of_gamut:
        gamut_warnings.append(f"Closest match: {result.closest_gamut.warning_message}")
    if gamut_warnings:
        st.warning("sRGB gamut warning:\n- " + "\n- ".join(gamut_warnings))

    if result.uncertainty.enabled:
        stability_cols = st.columns(4)
        stability_cols[0].metric("Monte Carlo Draws", f"{result.uncertainty.simulations_run}")
        stability_cols[1].metric("Repeat Rate", result.uncertainty.repeat_rate_label)
        stability_cols[2].metric("Stability", result.uncertainty.stability_score_label)
        stability_cols[3].metric("MC Delta-E Mean +/- SD", result.uncertainty.delta_band_label)
        st.caption(
            f"MC best-match Delta-E p05-p95: {result.uncertainty.percentile_band_label}. "
            f"Input sigma: {result.uncertainty_settings.sigma_label}."
        )
        if result.uncertainty.is_unstable:
            st.warning(result.uncertainty.message)
        else:
            st.info(result.uncertainty.message)

        if result.uncertainty.competitor_breakdown:
            with st.expander("Monte Carlo competitor frequencies", expanded=False):
                competitor_df = pd.DataFrame(result.uncertainty.competitor_breakdown).copy()
                competitor_df["Rate"] = competitor_df["Rate"].map(lambda x: f"{x * 100:.1f}%")
                st.dataframe(competitor_df, use_container_width=True, hide_index=True)

    cards = st.columns(2)
    with cards[0]:
        render_color_card("Input Color", "Your LAB input", result.input_rgb, result.input_lab)
    with cards[1]:
        render_color_card("Closest Match", result.closest_name, result.closest_rgb, result.closest_lab)

    render_blend_card(result.input_rgb, result.closest_rgb)

    top_matches = build_top_matches(
        dataset_df,
        result.input_lab,
        result.method,
        result.delta_e_preset,
        result.science_settings,
        top_n=10,
    )
    with st.expander("Top 10 nearest colors (sorted by Delta-E)", expanded=False):
        st.bar_chart(
            top_matches,
            x="Color Name",
            y="Delta E",
            color=THEME["accent"],
            horizontal=True,
            sort="Delta E",
            height=320,
        )

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
            ":sparkles: Comparison",
            ":bar_chart: LAB Bars",
            ":globe_with_meridians: 3D LAB",
            ":chart_with_upwards_trend: Delta-E",
            ":cyclone: Density",
            ":satellite: Scatter Matrix",
        ],
        default=":sparkles: Comparison",
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
    st.set_page_config(
        page_title="Getty Colour Identifier",
        layout="wide",
        page_icon="C",
        initial_sidebar_state="expanded",
    )
    inject_global_styles()
    suppress_key_overlay()

    (
        csv_file,
        rdf_file,
        input_lab,
        delta_e_method,
        delta_e_preset,
        science_settings,
        uncertainty_settings,
    ) = display_sidebar()
    method_label = get_delta_e_method_label(delta_e_method, delta_e_preset)

    input_preview_gamut: Optional[RgbGamutDiagnostics] = None
    if validate_lab_color(input_lab, report=False):
        input_rgb, input_preview_gamut = lab_to_rgb_with_diagnostics(
            input_lab,
            source_illuminant=science_settings.source_illuminant,
            target_illuminant=science_settings.target_illuminant,
            observer=science_settings.observer,
            chromatic_adaptation=science_settings.chromatic_adaptation,
        )
    else:
        input_rgb = (120, 120, 120)
    render_hero(input_rgb, input_lab, method_label, science_settings, input_preview_gamut)

    if csv_file is None:
        render_empty_state()
        return

    analyzer = ColorAnalyzer()
    with st.spinner("Loading dataset..."):
        if not analyzer.load_dataset(csv_file):
            st.error("Dataset loading failed. Please check the CSV file.")
            return

    render_status_pill("Dataset Ready")
    render_dataset_overview(analyzer.dataset_df, analyzer.dataset_quality_report)

    analyzer.load_rdf(rdf_file)
    analyzer.set_delta_e_method(delta_e_method)
    analyzer.set_delta_e_preset(delta_e_preset)
    analyzer.set_color_science_settings(science_settings)
    analyzer.set_uncertainty_settings(uncertainty_settings)

    render_input_preview(input_lab, input_rgb, method_label, science_settings, input_preview_gamut)

    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    analyze = st.button("Analyze Color", type="primary", use_container_width=True)
    if analyze:
        if analyzer.set_input_color(input_lab):
            with st.spinner("Analyzing..."):
                result = analyzer.match_color()
            if result:
                render_results_section(result, analyzer.rdf_alternatives_df, analyzer.dataset_df)
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
