"""
color_analyzer.py

An enhanced and modularized Streamlit-based LAB Color Analyzer.
This module provides a ColorAnalyzer class that encapsulates dataset loading,
RDF processing, color matching, visualization generation, and results display.

Author: Mochi the Code Catgirl (nyah~)
Refactored by: Your Friendly Research Assistant
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import re
from rdflib import Graph, URIRef, Namespace, Literal
from rdflib.namespace import RDF
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from typing import Union, List, Tuple, Optional, Any, IO

# =============================================================================
# Custom Exceptions
# =============================================================================

class InputError(Exception):
    """Exception raised for errors in the input LAB color."""
    pass

class DatasetError(Exception):
    """Exception raised for errors in the dataset file."""
    pass

class RDFParsingError(Exception):
    """Exception raised for errors during RDF file parsing."""
    pass

class ConversionError(Exception):
    """Exception raised for errors in color conversions."""
    pass

# =============================================================================
# Setup and Configuration
# =============================================================================

logging.basicConfig(
    filename='color_analyzer.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Compatibility for older NumPy versions
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item() if isinstance(a, np.ndarray) else a

REQUIRED_COLUMNS = {'L', 'A', 'B', 'Color Name'}

# =============================================================================
# Utility Functions (Validation, Conversion, Logging)
# =============================================================================

def _log_and_report(message: str, level: str = 'debug', error_type: Optional[str] = None) -> None:
    """Logs and reports messages to both the log file and Streamlit UI."""
    if level == 'debug':
        logger.debug(message)
    elif level == 'info':
        logger.info(message)
        st.info(message)
    elif level == 'warning':
        logger.warning(message)
        st.warning(message)
    elif level == 'error':
        logger.error(message)
        st.error(f"{error_type}: {message}" if error_type else message)

def validate_lab_color(lab: Union[List[float], Tuple[float, float, float], np.ndarray]) -> bool:
    """
    Validates that the LAB color input is a 3-element list/tuple/array with numerical values 
    within the proper ranges.
    """
    if not isinstance(lab, (list, tuple, np.ndarray)) or len(lab) != 3:
        _log_and_report("Input LAB color must be a list, tuple, or array of three numerical values.", 'error', 'Input Error')
        return False
    try:
        L, A, B = map(float, lab)
    except (ValueError, TypeError) as e:
        _log_and_report(f"LAB components must be numerical values. {e}", 'error', 'Input Error')
        return False
    if not (0 <= L <= 100 and -128 <= A <= 127 and -128 <= B <= 127):
        _log_and_report("LAB components are out of range. L: 0-100, A & B: -128 to 127.", 'error', 'Input Error')
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
        rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
        # Clip the values between 0 and 1, then scale to 255
        return tuple(int(max(0, min(1, c)) * 255) for c in [rgb.rgb_r, rgb.rgb_g, rgb.rgb_b])
    except Exception as e:
        _log_and_report(f"Error converting LAB to RGB: {e}", 'error', 'Conversion Error')
        return (0, 0, 0)

def format_rgb(rgb: Tuple[int, int, int]) -> str:
    """Formats an RGB tuple into a Plotly-friendly RGB string."""
    return f"rgb({rgb[0]},{rgb[1]},{rgb[2]})"

def calculate_delta_e(input_lab: Union[List[float], Tuple[float, float, float]],
                      dataset_df: pd.DataFrame,
                      method: str = 'euclidean') -> np.ndarray:
    """
    Calculates the Delta-E values between an input LAB color and all colors in the dataset.
    Supports Euclidean (ΔE76) and CIEDE2000 methods.
    """
    input_lab_arr = np.array(input_lab)
    if method.lower() == 'ciede2000':
        input_lab_obj = LabColor(*input_lab_arr)
        # Using list comprehension for clarity and potential speed-up
        delta_e_values = np.array([
            delta_e_cie2000(input_lab_obj, LabColor(*row))
            for row in dataset_df[['L', 'A', 'B']].values
        ])
    else:
        delta_e_values = np.linalg.norm(dataset_df[['L', 'A', 'B']].values - input_lab_arr, axis=1)
    return delta_e_values

def find_closest_color(input_lab: Union[List[float], Tuple[float, float, float]],
                       dataset_df: pd.DataFrame,
                       delta_e_method: str = 'euclidean') -> Tuple[Optional[pd.Series], Optional[float]]:
    """
    Finds the closest color from the dataset to the input LAB color based on Delta-E.
    Returns the closest color row and its Delta-E value.
    """
    if dataset_df.empty:
        _log_and_report("Dataset is empty after validation.", "error", "Dataset Error")
        return None, None
    delta_e_values = calculate_delta_e(input_lab, dataset_df, method=delta_e_method)
    if np.all(np.isnan(delta_e_values)):
        _log_and_report("Delta-E calculation resulted in all NaN values. Check dataset and input.", 'error')
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
    except Exception as e:
        _log_and_report(f"Error reading CSV file: {e}", 'error', 'File Reading Error')
        raise DatasetError("Failed to load dataset.")
    if not REQUIRED_COLUMNS.issubset(dataset_df.columns):
        missing_cols = REQUIRED_COLUMNS - set(dataset_df.columns)
        _log_and_report(f"CSV is missing required columns: {missing_cols}", 'error', 'Dataset Error')
        raise DatasetError("Dataset missing required columns.")
    dataset_df = dataset_df.copy()
    dataset_df['Color Name'] = dataset_df['Color Name'].astype(str).str.strip()
    for col in ['L', 'A', 'B']:
        dataset_df[col] = pd.to_numeric(dataset_df[col], errors='coerce')
    if dataset_df[list(REQUIRED_COLUMNS)].isnull().any().any():
        _log_and_report("CSV contains missing or non-numeric values in required columns.", 'error', 'Dataset Error')
        raise DatasetError("Dataset contains invalid values.")
    if dataset_df.empty:
        _log_and_report("CSV contains no valid rows after cleaning.", 'error', 'Dataset Error')
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
    except Exception as e:
        _log_and_report(f"Failed to parse RDF file: {e}", 'error', 'RDF Parsing Error')
        raise RDFParsingError("Failed to parse RDF file.")
    
    RDFS = Namespace("http://www.w3.org/2000/01/rdf-schema#")
    SKOS = Namespace("http://www.w3.org/2004/02/skos/core#")
    DC_ELEM = URIRef("http://purl.org/dc/elements/1.1/description")
    
    subjects = list(g.subjects(predicate=RDF.type, object=URIRef("http://www.w3.org/2004/02/skos/core#Concept")))
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
                if re.search(r'\d', term_text) or len(term_text) > 50:
                    return
            else:
                if len(term_text) > 100:
                    return
            rows.append({"Term": term_text, "Language": term.language or "unknown"})
    
    # Process each subject
    for subj in subjects:
        for label in g.objects(subject=subj, predicate=RDFS.label):
            add_term(label, skip_digit_filter=False)
        for alt in g.objects(subject=subj, predicate=SKOS.altLabel):
            add_term(alt, skip_digit_filter=False)
    
    # Process DC description terms (not tied to a specific subject)
    for desc in g.objects(predicate=DC_ELEM):
        add_term(desc, skip_digit_filter=True)
    
    return pd.DataFrame(rows).drop_duplicates()

# =============================================================================
# Visualization Functions
# =============================================================================

def truncate_label(label: str, max_length: int = 20) -> str:
    """Truncates a label to a maximum length."""
    return label if len(label) <= max_length else label[:max_length] + "..."

def create_alternative_terms_sunburst(df: pd.DataFrame, base_color: Optional[str] = None,
                                        trunc_length: int = 20) -> go.Figure:
    """
    Generates a sunburst chart from the alternative color terms DataFrame.
    Truncates overly long labels for display while showing the full text in the hover tooltip.
    """
    if df.empty:
        return go.Figure()
    
    # Choose the main term: prefer English labels if available.
    english_terms = df[df["Language"] == "en"]
    main_term = english_terms["Term"].iloc[0] if not english_terms.empty else df["Term"].iloc[0]
    
    df_alt = df[df["Term"] != main_term].copy()
    data = [{'id': 'root', 'parent': '', 'name': main_term}]
    
    # Create a branch for each language.
    for lang in df_alt["Language"].unique():
        data.append({'id': f'lang_{lang}', 'parent': 'root', 'name': lang})
    
    # Add each alternative term.
    for idx, row in df_alt.iterrows():
        data.append({'id': str(idx), 'parent': f'lang_{row["Language"]}', 'name': row["Term"]})
    
    sunburst_df = pd.DataFrame(data)
    sunburst_df["full_name"] = sunburst_df["name"]
    sunburst_df["name"] = sunburst_df["name"].apply(lambda x: truncate_label(x, trunc_length))
    
    fig = px.sunburst(
        sunburst_df,
        ids="id",
        names="name",
        parents="parent",
        custom_data=["full_name"],
        title="Alternative Terms Sunburst Chart",
        template="plotly_white",
        color="id" if not base_color else None,
        color_discrete_sequence=[base_color] if base_color else None,
    )
    fig.update_traces(
        textfont=dict(size=18),
        insidetextorientation='radial',
        hovertemplate='<b>%{customdata[0]}</b><extra></extra>'
    )
    fig.update_layout(
        title_font=dict(size=24),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_color_comparison_plot(input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int],
                                 input_lab: List[float], closest_lab: List[float],
                                 closest_color_name: str, delta_e: float) -> go.Figure:
    """
    Creates a side-by-side scatter plot comparing the input color and the closest matched color.
    """
    fig = go.Figure(data=[
        go.Scatter(
            x=[0], y=[1],
            mode='markers',
            marker=dict(
                size=50,
                color=format_rgb(input_rgb),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name='Input Color',
            hovertemplate=(f"Input LAB: L={input_lab[0]:.2f}, A={input_lab[1]:.2f}, B={input_lab[2]:.2f}"
                           f"<br>Delta-E: {delta_e:.2f}<extra></extra>")
        ),
        go.Scatter(
            x=[1], y=[1],
            mode='markers',
            marker=dict(
                size=50,
                color=format_rgb(closest_rgb),
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=f'Closest: {closest_color_name}',
            hovertemplate=(f"Closest LAB: L={closest_lab[0]:.2f}, A={closest_lab[1]:.2f}, B={closest_lab[2]:.2f}"
                           f"<extra></extra>")
        )
    ])
    fig.update_layout(
        title='Input Color vs Closest ISCC-NBS Color',
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[-0.5, 1.5]),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False, range=[0.5, 1.5]),
        template='plotly_white',
        margin=dict(l=50, r=50, t=80, b=50),
        showlegend=False,
        annotations=[
            dict(x=0, y=1.55, text='Input Color', showarrow=False,
                 font=dict(size=14, color='black'), xanchor='center'),
            dict(x=1, y=1.55, text=f'Closest: {closest_color_name}', showarrow=False,
                 font=dict(size=14, color='black'), xanchor='center')
        ]
    )
    return fig

def create_lab_comparison_bar(input_lab: List[float], closest_lab: List[float],
                              closest_color_name: str, input_rgb: Tuple[int, int, int],
                              closest_rgb: Tuple[int, int, int]) -> go.Figure:
    """
    Creates a bar chart comparing the LAB components of the input color and the closest matched color.
    """
    components = ['L', 'A', 'B']
    data = pd.DataFrame({
        'Component': components * 2,
        'Value': input_lab + closest_lab,
        'Type': ['Input LAB'] * 3 + [f'Closest LAB: {closest_color_name}'] * 3
    })
    color_map = {
        'Input LAB': format_rgb(input_rgb),
        f'Closest LAB: {closest_color_name}': format_rgb(closest_rgb)
    }
    fig = px.bar(
        data_frame=data,
        x='Component',
        y='Value',
        color='Type',
        barmode='group',
        hover_data=['Value'],
        title='LAB Value Comparison',
        template='plotly_white',
        color_discrete_map=color_map
    )
    for i, component in enumerate(components):
        delta = abs(input_lab[i] - closest_lab[i])
        fig.add_annotation(
            x=component,
            y=max(input_lab[i], closest_lab[i]) + 5,
            text=f'Delta: {delta:.2f}',
            showarrow=False,
            font=dict(size=12, color='black')
        )
    fig.update_layout(
        xaxis_title='LAB Components',
        yaxis_title='Values',
        legend_title='Color Type',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_3d_lab_plot(input_lab: List[float], closest_lab: List[float],
                       closest_color_name: str, dataset_df: pd.DataFrame,
                       input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int]) -> go.Figure:
    """
    Generates a 3D scatter plot of the LAB color space, marking the dataset colors,
    the input color, and the closest color.
    """
    dataset_points = go.Scatter3d(
        x=dataset_df['L'], y=dataset_df['A'], z=dataset_df['B'],
        mode='markers',
        marker=dict(size=3, color='lightgrey', opacity=0.5),
        name='Dataset Colors',
        hoverinfo='text',
        text=dataset_df['Color Name']
    )
    input_point = go.Scatter3d(
        x=[input_lab[0]], y=[input_lab[1]], z=[input_lab[2]],
        mode='markers+text',
        marker=dict(size=10, color=format_rgb(input_rgb), opacity=1),
        text=['Input Color'],
        textposition='top center',
        name='Input Color',
        hoverinfo='text'
    )
    closest_point = go.Scatter3d(
        x=[closest_lab[0]], y=[closest_lab[1]], z=[closest_lab[2]],
        mode='markers+text',
        marker=dict(size=10, color=format_rgb(closest_rgb), opacity=1),
        text=[f'Closest: {closest_color_name}'],
        textposition='top center',
        name='Closest Color',
        hoverinfo='text'
    )
    fig = go.Figure(data=[dataset_points, input_point, closest_point])
    fig.update_layout(
        title='3D LAB Color Space Visualization',
        scene=dict(
            xaxis_title='L', yaxis_title='A', zaxis_title='B',
            xaxis=dict(range=[0, 100], backgroundcolor='white'),
            yaxis=dict(range=[-128, 127], backgroundcolor='white'),
            zaxis=dict(range=[-128, 127], backgroundcolor='white'),
            bgcolor='white',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        legend=dict(x=0.7, y=0.9, bgcolor='rgba(255,255,255,1)', bordercolor='rgba(0,0,0,0)'),
        template='plotly_white',
        margin=dict(l=0, r=0, t=80, b=0)
    )
    return fig

def create_delta_e_histogram(delta_e_values: np.ndarray) -> go.Figure:
    """
    Creates a histogram of Delta-E values across the dataset.
    """
    fig = px.histogram(
        x=delta_e_values,
        nbins=30,
        title='Delta-E Distribution',
        labels={'x': 'Delta-E Value', 'y': 'Count'},
        template='plotly_white',
        opacity=0.75
    )
    fig.update_layout(
        xaxis=dict(title='Delta-E'),
        yaxis=dict(title='Frequency'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_color_density_heatmap(dataset_df: pd.DataFrame) -> go.Figure:
    """
    Generates a density heatmap in the A-B plane from the dataset LAB values.
    """
    fig = px.density_heatmap(
        dataset_df,
        x='A',
        y='B',
        nbinsx=50,
        nbinsy=50,
        title='Color Density Heatmap in A-B Plane',
        labels={'A': 'A Component', 'B': 'B Component'},
        color_continuous_scale='Viridis',
        template='plotly_white'
    )
    fig.update_layout(
        xaxis_title='A',
        yaxis_title='B',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

@st.cache_data
def cached_lab_to_rgb(l: float, a: float, b: float) -> Tuple[int, int, int]:
    """Cached helper for converting LAB to RGB."""
    return lab_to_rgb([l, a, b])

def create_pairwise_scatter_matrix(dataset_df: pd.DataFrame, input_lab: List[float],
                                   closest_lab: List[float]) -> go.Figure:
    """
    Creates a scatter matrix plot showing pairwise relationships of LAB values,
    including the input and closest color.
    """
    splom_df = dataset_df.copy()
    input_row = {'L': input_lab[0], 'A': input_lab[1], 'B': input_lab[2], 'Color Name': 'Input Color'}
    closest_row = {'L': closest_lab[0], 'A': closest_lab[1], 'B': closest_lab[2], 'Color Name': 'Closest Color'}
    splom_df = pd.concat([splom_df, pd.DataFrame([input_row, closest_row])], ignore_index=True)
    
    splom_df['Color Group'] = splom_df.apply(lambda row: format_rgb(cached_lab_to_rgb(row["L"], row["A"], row["B"])), axis=1)
    
    splom_trace = go.Splom(
        dimensions=[
            dict(label='L', values=splom_df['L']),
            dict(label='A', values=splom_df['A']),
            dict(label='B', values=splom_df['B'])
        ],
        text=splom_df['Color Name'],
        marker=dict(size=5, color=splom_df['Color Group'], opacity=0.7),
        diagonal_visible=False,
        showupperhalf=False,
        name='Colors'
    )
    fig_splom = go.Figure(data=[splom_trace])
    fig_splom.update_layout(
        title='Pairwise LAB Relationships',
        template='plotly_white',
        dragmode='select',
        height=800
    )
    return fig_splom

def display_results_table(results: dict) -> None:
    """
    Displays the results in a formatted HTML table using Streamlit.
    """
    df = pd.DataFrame([results])
    def color_rgb_html(rgb_tuple: Tuple[int, int, int]) -> str:
        return f'<div style="background-color:{format_rgb(rgb_tuple)}; width:100px; height:20px;"></div>'
    df['Input RGB'] = df['Input RGB'].apply(color_rgb_html)
    df['Closest RGB'] = df['Closest RGB'].apply(color_rgb_html)
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# =============================================================================
# ColorAnalyzer Class
# =============================================================================

class ColorAnalyzer:
    """
    A class-based implementation of the LAB Color Analyzer.
    Encapsulates dataset loading, RDF processing, color matching, visualization, and result display.
    """
    REQUIRED_COLUMNS = REQUIRED_COLUMNS  # Class variable for configuration

    def __init__(self) -> None:
        self.dataset_df: Optional[pd.DataFrame] = None
        self.rdf_alternatives_df: Optional[pd.DataFrame] = None
        self.input_lab: Optional[List[float]] = None
        self.delta_e_method: str = 'euclidean'
        self.closest_color: Optional[pd.Series] = None
        self.delta_e: Optional[float] = None

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
        if validate_lab_color(lab):
            self.input_lab = lab
            return True
        return False

    def set_delta_e_method(self, method: str) -> None:
        """Sets the Delta-E calculation method."""
        self.delta_e_method = method

    def match_color(self) -> bool:
        """Finds the closest color from the dataset to the input LAB color."""
        if self.dataset_df is None or self.input_lab is None:
            _log_and_report("Dataset or input LAB color not set.", 'error', 'Processing Error')
            return False
        self.closest_color, self.delta_e = find_closest_color(self.input_lab, self.dataset_df, delta_e_method=self.delta_e_method)
        if self.closest_color is None:
            return False
        return True

    def generate_visualizations(self) -> dict:
        """
        Generates and returns a dictionary of Plotly figures for various visualizations.
        Returns keys: comparison, lab_bar, lab_3d, delta_hist, density_heatmap, scatter_matrix.
        """
        if self.dataset_df is None or self.input_lab is None or self.closest_color is None:
            _log_and_report("Missing data for visualization.", 'error', 'Visualization Error')
            return {}
        closest_lab = [self.closest_color['L'], self.closest_color['A'], self.closest_color['B']]
        input_rgb = lab_to_rgb(self.input_lab)
        closest_rgb = lab_to_rgb(closest_lab)
        closest_color_name = self.closest_color['Color Name']

        figs = {
            "comparison": create_color_comparison_plot(input_rgb, closest_rgb, self.input_lab, closest_lab, closest_color_name, self.delta_e),
            "lab_bar": create_lab_comparison_bar(self.input_lab, closest_lab, closest_color_name, input_rgb, closest_rgb),
            "lab_3d": create_3d_lab_plot(self.input_lab, closest_lab, closest_color_name, self.dataset_df, input_rgb, closest_rgb),
            "delta_hist": create_delta_e_histogram(calculate_delta_e(self.input_lab, self.dataset_df, method=self.delta_e_method)),
            "density_heatmap": create_color_density_heatmap(self.dataset_df),
            "scatter_matrix": create_pairwise_scatter_matrix(self.dataset_df, self.input_lab, closest_lab)
        }
        return figs

    def display_results(self) -> None:
        """Displays the results table and provides a download button for CSV export."""
        if self.dataset_df is None or self.input_lab is None or self.closest_color is None:
            st.error("Missing results data.")
            return
        closest_lab = [self.closest_color['L'], self.closest_color['A'], self.closest_color['B']]
        input_rgb = lab_to_rgb(self.input_lab)
        closest_rgb = lab_to_rgb(closest_lab)
        closest_color_name = self.closest_color['Color Name']
        st.markdown("### **Results:**")
        st.markdown(
            f"""
            | Metric                     | Value                                                        |
            |----------------------------|--------------------------------------------------------------|
            | **Input LAB Color**        | L={self.input_lab[0]:.2f}, A={self.input_lab[1]:.2f}, B={self.input_lab[2]:.2f} |
            | **Closest ISCC-NBS Color** | {closest_color_name}                                         |
            | **Delta-E Value ({self.delta_e_method.upper()})** | {self.delta_e:.2f}                                                |
            | **Closest LAB Color**      | L={closest_lab[0]:.2f}, A={closest_lab[1]:.2f}, B={closest_lab[2]:.2f} |
            | **Input RGB Color**        | <span style="background-color:{format_rgb(input_rgb)}; padding: 5px; border: 1px solid black;">   </span> |
            | **Closest RGB Color**      | <span style="background-color:{format_rgb(closest_rgb)}; padding: 5px; border: 1px solid black;">   </span> |
            """,
            unsafe_allow_html=True
        )
        # Display alternative terms if available
        if self.rdf_alternatives_df is not None and not self.rdf_alternatives_df.empty:
            st.markdown("### **Alternative Color Terms from RDF:**")
            st.dataframe(self.rdf_alternatives_df)
            base_color = format_rgb(closest_rgb)
            fig_sunburst = create_alternative_terms_sunburst(self.rdf_alternatives_df, base_color=base_color)
            st.plotly_chart(fig_sunburst, use_container_width=True)
        results = {
            'Input LAB': f"L={self.input_lab[0]:.2f}, A={self.input_lab[1]:.2f}, B={self.input_lab[2]:.2f}",
            'Closest ISCC-NBS Color': closest_color_name,
            'Delta-E Value': f"{self.delta_e:.2f}",
            'Closest LAB': f"L={closest_lab[0]:.2f}, A={closest_lab[1]:.2f}, B={closest_lab[2]:.2f}",
            'Input RGB': input_rgb,
            'Closest RGB': closest_rgb
        }
        display_results_table(results)
        csv_results = pd.DataFrame([results]).to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_results,
            file_name='color_match_results.csv',
            mime='text/csv',
        )

# =============================================================================
# Streamlit UI Functions
# =============================================================================

def display_sidebar() -> Tuple[Any, Any, List[float], str]:
    """
    Displays the sidebar widgets for file uploads and LAB color input.
    Returns:
        - CSV file uploader object.
        - RDF file uploader object.
        - Input LAB color list.
        - Delta-E method as a lowercase string ('euclidean' or 'ciede2000').
    """
    st.sidebar.header("Upload & Input")
    csv_file = st.sidebar.file_uploader("Upload 'iscc_nbs_lab_colors.csv'", type=['csv'])
    rdf_file = st.sidebar.file_uploader("Upload Getty AAT RDF (XML)", type=['xml'])
    with st.sidebar.expander("Instructions", expanded=False):
        st.markdown(
            """
            1. **Upload Dataset:** Upload the CSV file containing LAB values and color names.
            2. **Upload RDF (Optional):** Upload the RDF file with Getty AAT color terms.
            3. **Input LAB Color:** Use sliders or manual input for LAB values.
            4. **Choose Delta-E Metric:** Select Euclidean ΔE76 or CIEDE2000.
            5. **Find Closest Color:** Click to analyze and visualize.
            """
        )
    delta_e_metric = st.sidebar.radio("Select Delta-E metric:", ("Euclidean ΔE76", "CIEDE2000"), index=0)
    st.sidebar.markdown("### LAB Color Input")
    input_method = st.sidebar.radio("Input Method:", ("Slider Input", "Manual Input"), horizontal=True)
    if input_method == "Slider Input":
        lab_l = st.sidebar.slider("L:", 0.0, 100.0, 50.0, 0.1)
        lab_a = st.sidebar.slider("A:", -128.0, 127.0, 0.0, 0.1)
        lab_b = st.sidebar.slider("B:", -128.0, 127.0, 0.0, 0.1)
    else:
        lab_l = st.sidebar.number_input("L (0-100):", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        lab_a = st.sidebar.number_input("A (-128 to 127):", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)
        lab_b = st.sidebar.number_input("B (-128 to 127):", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)
    input_lab = [lab_l, lab_a, lab_b]
    # Return the lowercase method string ('euclidean' or 'ciede2000')
    method_key = delta_e_metric.lower().split()[0]
    return csv_file, rdf_file, input_lab, method_key

def main() -> None:
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Enhanced LAB Color Analyzer", layout="wide", page_icon="🎨")
    st.markdown("<style>body { background-color: white; color: black; }</style>", unsafe_allow_html=True)
    st.title("Enhanced LAB Color Analyzer")
    st.markdown(
        """
        Welcome to the **Enhanced LAB Color Analyzer**!  
        Upload your ISCC-NBS LAB colors dataset and input your LAB color values to find the closest 
        matching color. View detailed visualizations and, optionally, explore alternative color terms 
        from a Getty AAT RDF file.
        """
    )

    # Sidebar: file uploads and input
    csv_file, rdf_file, input_lab, delta_e_method = display_sidebar()
    if csv_file is None:
        st.info("Please upload your 'iscc_nbs_lab_colors.csv' file to begin.")
        return

    analyzer = ColorAnalyzer()
    with st.spinner("Loading dataset..."):
        if not analyzer.load_dataset(csv_file):
            st.error("Dataset loading failed. Please check the CSV file.")
            return
    st.success("Dataset uploaded and validated successfully.")
    with st.expander("View Dataset Preview", expanded=False):
        st.dataframe(analyzer.dataset_df.head())

    analyzer.load_rdf(rdf_file)
    analyzer.set_delta_e_method(delta_e_method)
    if st.sidebar.button("Find Closest Color"):
        if analyzer.set_input_color(input_lab):
            with st.spinner("Processing..."):
                if analyzer.match_color():
                    analyzer.display_results()
                    figs = analyzer.generate_visualizations()
                    if figs:
                        tabs = st.tabs(["Color Comparison", "LAB Comparison", "3D LAB Plot",
                                         "Delta-E Histogram", "Color Density", "Pairwise Scatter"])
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
                    else:
                        st.error("Error generating visualizations.")
                else:
                    st.error("An error occurred during color matching. Please check your inputs and dataset.")
        else:
            st.error("Invalid LAB color input. Please check the values.")

if __name__ == "__main__":
    main()
