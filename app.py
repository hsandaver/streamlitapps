import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import logging
import json
import networkx as nx  # Added for network graph creation
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000
from typing import Union, List, Tuple, Optional, Any
from io import StringIO

# Monkey-patch numpy.asscalar for compatibility with colormath (numpy.asscalar is deprecated in newer versions)
if not hasattr(np, 'asscalar'):
    np.asscalar = lambda a: a.item() if isinstance(a, np.ndarray) else a

# =============================================================================
# Debugging & Logging Setup
# =============================================================================
logging.basicConfig(
    filename='debug.log',
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

def debug_log(message: str) -> None:
    logging.debug(message)

# =============================================================================
# Utility Functions
# =============================================================================
def validate_lab_color(lab: Union[List[float], Tuple[float, float, float], np.ndarray]) -> bool:
    """
    Validate that the LAB color is a list, tuple, or numpy array of three numerical values.
    
    Returns:
        bool: True if valid, False otherwise.
    """
    debug_log(f"Validating LAB color: {lab}")
    if not isinstance(lab, (list, tuple, np.ndarray)) or len(lab) != 3:
        st.error("Input LAB color must be a list, tuple, or array of three numerical values.")
        debug_log("Validation failed: Incorrect type or length")
        return False

    try:
        L, A, B = float(lab[0]), float(lab[1]), float(lab[2])
    except (ValueError, TypeError) as e:
        st.error("LAB components must be numerical values.")
        debug_log(f"Validation failed: LAB conversion error - {e}")
        return False

    if not (0 <= L <= 100):
        st.error("L component must be between 0 and 100.")
        debug_log("Validation failed: L component out of range")
        return False
    if not (-128 <= A <= 127):
        st.error("A component must be between -128 and 127.")
        debug_log("Validation failed: A component out of range")
        return False
    if not (-128 <= B <= 127):
        st.error("B component must be between -128 and 127.")
        debug_log("Validation failed: B component out of range")
        return False

    debug_log("LAB color validation passed")
    return True

@st.cache_data(show_spinner=False)
def lab_to_rgb(lab_color: Union[List[float], Tuple[float, float, float], np.ndarray]) -> Tuple[int, int, int]:
    """
    Convert a LAB color to its RGB representation.
    
    Returns:
        Tuple[int, int, int]: RGB color values clamped between 0 and 255.
    """
    debug_log(f"Converting LAB to RGB for: {lab_color}")
    try:
        lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
        rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
        rgb_clamped = (
            int(max(0, min(rgb.rgb_r, 1)) * 255),
            int(max(0, min(rgb.rgb_g, 1)) * 255),
            int(max(0, min(rgb.rgb_b, 1)) * 255)
        )
        debug_log(f"Converted LAB {lab_color} to RGB {rgb_clamped}")
        return rgb_clamped
    except Exception as e:
        st.error(f"Error converting LAB to RGB: {e}")
        debug_log(f"Error converting LAB to RGB for {lab_color}: {e}")
        return (0, 0, 0)

def calculate_delta_e_euclidean(input_lab: Union[List[float], Tuple[float, float, float]], 
                                dataset_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Delta-E using the simple Euclidean distance (ΔE*76).
    """
    debug_log("Calculating Euclidean Delta-E (ΔE*76).")
    input_lab_arr = np.array(input_lab)
    delta_e = np.linalg.norm(dataset_df[['L', 'A', 'B']].values - input_lab_arr, axis=1)
    debug_log("Euclidean Delta-E calculation complete.")
    return delta_e

def calculate_delta_e_ciede2000(input_lab: Union[List[float], Tuple[float, float, float]], 
                                dataset_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate Delta-E using the CIEDE2000 formula.
    """
    debug_log("Calculating CIEDE2000 Delta-E.")
    input_lab_obj = LabColor(lab_l=input_lab[0], lab_a=input_lab[1], lab_b=input_lab[2])
    delta_e_list = []
    for idx, row in dataset_df.iterrows():
        lab_obj = LabColor(lab_l=row['L'], lab_a=row['A'], lab_b=row['B'])
        delta_e = delta_e_cie2000(input_lab_obj, lab_obj)
        delta_e_list.append(delta_e)
    debug_log("CIEDE2000 Delta-E calculation complete.")
    return np.array(delta_e_list)

def find_closest_color(input_lab: Union[List[float], Tuple[float, float, float]], 
                       dataset_df: pd.DataFrame,
                       delta_e_func=calculate_delta_e_euclidean) -> Tuple[Optional[pd.Series], Optional[float]]:
    """
    Find the closest matching color from the dataset based on the Delta-E distance.
    
    Returns:
        Tuple containing the closest color row and its Delta-E value.
    """
    debug_log(f"Finding closest color for input LAB: {input_lab}")
    delta_e_values = delta_e_func(input_lab, dataset_df)
    if np.all(np.isnan(delta_e_values)):
        st.error("Delta-E calculation resulted in all NaN values. Check your dataset and input LAB values.")
        debug_log("All Delta-E values are NaN.")
        return None, None
    min_idx = np.nanargmin(delta_e_values)
    min_delta_e = delta_e_values[min_idx]
    closest_color = dataset_df.iloc[min_idx]
    debug_log(f"Closest color found: {closest_color['Color Name']} with Delta-E: {min_delta_e}")
    return closest_color, min_delta_e

# =============================================================================
# Linked Data Enhancements
# =============================================================================
@st.cache_data(show_spinner=False)
def load_linked_data(linked_data_file: Any) -> dict:
    """
    Load Getty Linked Data JSON.
    
    Returns:
        dict: The parsed JSON data.
    """
    debug_log("Loading Linked Data from JSON file.")
    try:
        linked_data = json.load(linked_data_file)
        debug_log("Linked Data loaded successfully.")
        return linked_data
    except Exception as e:
        st.error(f"Error loading Linked Data JSON file: {e}")
        debug_log(f"Error loading Linked Data JSON file: {e}")
        return {}

def create_semantic_relationship_graph(linked_data: dict) -> go.Figure:
    """
    Build and return a network graph (using Plotly) that visualizes semantic relationships
    from the Getty JSON.
    """
    G = nx.DiGraph()
    central_id = linked_data.get("id", "central")
    central_label = linked_data.get("_label", "Central")
    G.add_node(central_id, label=central_label)
    
    # Define keys in the JSON that represent semantic relationships
    relationship_keys = ["broader", "rdfs:seeAlso", "subject_of", "la:related_from_by", "skos:inScheme"]
    
    for key in relationship_keys:
        if key in linked_data:
            value = linked_data[key]
            if isinstance(value, list):
                for item in value:
                    if key == "la:related_from_by":
                        target_id = item.get("la:relates_to", {}).get("id")
                        target_label = target_id  # No explicit label provided
                    else:
                        target_id = item.get("id")
                        target_label = item.get("_label") or item.get("content") or target_id
                    if target_id:
                        G.add_node(target_id, label=target_label)
                        G.add_edge(central_id, target_id, relationship=key)
            elif isinstance(value, dict):
                target_id = value.get("id")
                target_label = value.get("_label") or value.get("content") or target_id
                if target_id:
                    G.add_node(target_id, label=target_label)
                    G.add_edge(central_id, target_id, relationship=key)
    
    # Compute layout for nodes
    pos = nx.spring_layout(G)
    
    # Build edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines'
    )
    
    # Build node traces
    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[1].get("label", node[0]))
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color='#FFA500',
            size=20,
            line_width=2
        )
    )
    
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Semantic Relationships",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[dict(
                            text="Semantic network of Getty AAT term relationships",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002
                        )],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    debug_log("Semantic relationship graph created successfully.")
    return fig

# =============================================================================
# Visualization Functions (Other)
# =============================================================================
def create_color_comparison_plot(input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int],
                                 input_lab: List[float], closest_lab: List[float],
                                 closest_color_name: str, delta_e: float) -> go.Figure:
    debug_log("Creating color comparison plot.")
    data = pd.DataFrame({
        'Color Type': ['Input Color', f'Closest: {closest_color_name}'],
        'RGB': [f'rgb{input_rgb}', f'rgb{closest_rgb}'],
        'LAB': [f"L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}",
                f"L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}"],
        'Delta-E': [delta_e, 'N/A']
    })

    fig = px.scatter(
        data_frame=data,
        x=[0, 1],
        y=[1, 1],
        color='RGB',
        hover_data=['Color Type', 'LAB', 'Delta-E'],
        labels={'x': '', 'y': ''},
        title='Input Color vs Closest ISCC-NBS Color'
    )
    fig.update_traces(marker=dict(size=50, line=dict(width=2, color='DarkSlateGrey')))
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        template='plotly_dark',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    fig.add_annotation(
        x=0, y=1.05, text='Input Color', showarrow=False,
        font=dict(size=14, color='white'), xanchor='center'
    )
    fig.add_annotation(
        x=1, y=1.05, text=f'Closest: {closest_color_name}', showarrow=False,
        font=dict(size=14, color='white'), xanchor='center'
    )
    debug_log("Color comparison plot created successfully.")
    return fig

def create_lab_comparison_bar(input_lab: List[float], closest_lab: List[float],
                              closest_color_name: str, input_rgb: Tuple[int, int, int],
                              closest_rgb: Tuple[int, int, int]) -> go.Figure:
    debug_log("Creating LAB comparison bar plot.")
    components = ['L', 'A', 'B']
    data = pd.DataFrame({
        'Component': components * 2,
        'Value': input_lab + closest_lab,
        'Type': ['Input LAB'] * 3 + [f'Closest LAB: {closest_color_name}'] * 3
    })
    color_map = {
        'Input LAB': f'rgb{input_rgb}',
        f'Closest LAB: {closest_color_name}': f'rgb{closest_rgb}'
    }
    fig = px.bar(
        data_frame=data,
        x='Component',
        y='Value',
        color='Type',
        barmode='group',
        hover_data=['Value'],
        title='LAB Value Comparison',
        template='plotly_dark',
        color_discrete_map=color_map
    )
    for i, component in enumerate(components):
        delta = abs(input_lab[i] - closest_lab[i])
        fig.add_annotation(
            x=component,
            y=max(input_lab[i], closest_lab[i]) + 5,
            text=f'Delta: {delta:.2f}',
            showarrow=False,
            font=dict(size=12, color='white')
        )
    fig.update_layout(
        xaxis_title='LAB Components',
        yaxis_title='Values',
        legend_title='Color Type',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    debug_log("LAB comparison bar plot created successfully.")
    return fig

def create_3d_lab_plot(input_lab: List[float], closest_lab: List[float],
                       closest_color_name: str, dataset_df: pd.DataFrame,
                       input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int]) -> go.Figure:
    debug_log("Creating 3D LAB color space plot.")
    dataset_points = go.Scatter3d(
        x=dataset_df['L'],
        y=dataset_df['A'],
        z=dataset_df['B'],
        mode='markers',
        marker=dict(size=3, color='lightgrey', opacity=0.5),
        name='Dataset Colors',
        hoverinfo='text',
        text=dataset_df['Color Name']
    )
    input_rgb_str = f'rgb{input_rgb}'
    closest_rgb_str = f'rgb{closest_rgb}'
    input_point = go.Scatter3d(
        x=[input_lab[0]],
        y=[input_lab[1]],
        z=[input_lab[2]],
        mode='markers+text',
        marker=dict(size=10, color=input_rgb_str, opacity=1),
        text=['Input Color'],
        textposition='top center',
        name='Input Color',
        hoverinfo='text'
    )
    closest_point = go.Scatter3d(
        x=[closest_lab[0]],
        y=[closest_lab[1]],
        z=[closest_lab[2]],
        mode='markers+text',
        marker=dict(size=10, color=closest_rgb_str, opacity=1),
        text=[f'Closest: {closest_color_name}'],
        textposition='top center',
        name='Closest Color',
        hoverinfo='text'
    )
    fig = go.Figure(data=[dataset_points, input_point, closest_point])
    fig.update_layout(
        title='3D LAB Color Space Visualization',
        scene=dict(
            xaxis_title='L',
            yaxis_title='A',
            zaxis_title='B',
            xaxis=dict(range=[0, 100], backgroundcolor='rgb(20, 20, 20)'),
            yaxis=dict(range=[-128, 127], backgroundcolor='rgb(20, 20, 20)'),
            zaxis=dict(range=[-128, 127], backgroundcolor='rgb(20, 20, 20)'),
            bgcolor='rgb(20, 20, 20)',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        legend=dict(x=0.7, y=0.9, bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,0,0,0)'),
        template='plotly_dark',
        margin=dict(l=0, r=0, t=80, b=0)
    )
    debug_log("3D LAB color space plot created successfully.")
    return fig

def create_delta_e_histogram(delta_e_values: np.ndarray) -> go.Figure:
    debug_log("Creating Delta-E histogram.")
    fig = px.histogram(
        x=delta_e_values,
        nbins=30,
        title='Delta-E Distribution',
        labels={'x': 'Delta-E Value', 'y': 'Count'},
        template='plotly_dark',
        opacity=0.75
    )
    fig.update_layout(
        xaxis=dict(title='Delta-E'),
        yaxis=dict(title='Frequency'),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    debug_log("Delta-E histogram created successfully.")
    return fig

def create_color_density_heatmap(dataset_df: pd.DataFrame) -> go.Figure:
    debug_log("Creating color density heatmap.")
    fig = px.density_heatmap(
        dataset_df,
        x='A',
        y='B',
        nbinsx=50,
        nbinsy=50,
        title='Color Density Heatmap in A-B Plane',
        labels={'A': 'A Component', 'B': 'B Component'},
        color_continuous_scale='Viridis',
        template='plotly_dark'
    )
    fig.update_layout(
        xaxis_title='A',
        yaxis_title='B',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    debug_log("Color density heatmap created successfully.")
    return fig

def create_pairwise_scatter_matrix(dataset_df: pd.DataFrame, input_lab: List[float],
                                   closest_lab: List[float]) -> go.Figure:
    debug_log("Creating pairwise scatter matrix.")
    splom_df = dataset_df.copy()
    input_row = {'L': input_lab[0], 'A': input_lab[1], 'B': input_lab[2], 'Color Name': 'Input Color'}
    closest_row = {'L': closest_lab[0], 'A': closest_lab[1], 'B': closest_lab[2], 'Color Name': 'Closest Color'}
    splom_df = pd.concat([splom_df, pd.DataFrame([input_row, closest_row])], ignore_index=True)

    cache_rgb = {}
    def map_color(color_name: str, lab_values: List[float]) -> str:
        key = tuple(lab_values)
        if key not in cache_rgb:
            cache_rgb[key] = f'rgb{lab_to_rgb(lab_values)}'
        return cache_rgb[key]

    splom_df['Color Group'] = splom_df.apply(lambda row: map_color(row['Color Name'], [row['L'], row['A'], row['B']]), axis=1)
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
        template='plotly_dark',
        dragmode='select',
        height=800
    )
    debug_log("Pairwise scatter matrix created successfully.")
    return fig_splom

def display_results_table(results: dict) -> None:
    debug_log("Displaying results table.")
    df = pd.DataFrame([results])

    def color_rgb(cell: str) -> str:
        return f'<div style="background-color:{cell}; width:100px; height:20px;"></div>'

    df['Input RGB'] = df['Input RGB'].apply(lambda x: color_rgb(f'rgb{x[0]}, {x[1]}, {x[2]}'))
    df['Closest RGB'] = df['Closest RGB'].apply(lambda x: color_rgb(f'rgb{x[0]}, {x[1]}, {x[2]}'))
    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)
    debug_log("Results table displayed.")

@st.cache_data(show_spinner=True)
def load_dataset(uploaded_file: Any) -> pd.DataFrame:
    debug_log("Loading dataset from uploaded file.")
    try:
        dataset_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Error reading CSV file: {e}")
        debug_log(f"Error reading CSV file: {e}")
        return pd.DataFrame()
        
    required_columns = {'L', 'A', 'B', 'Color Name'}
    if not required_columns.issubset(set(dataset_df.columns)):
        missing = required_columns - set(dataset_df.columns)
        st.error(f"CSV file is missing required columns: {missing}")
        debug_log(f"Dataset validation failed: Missing columns {missing}")
        return pd.DataFrame()
    if dataset_df[list(required_columns)].isnull().any().any():
        st.error("CSV file contains missing values in required columns.")
        debug_log("Dataset validation failed: Missing values detected.")
        return pd.DataFrame()
    debug_log("Dataset loaded and validated successfully.")
    return dataset_df

# =============================================================================
# Main Application
# =============================================================================
def main() -> None:
    st.set_page_config(page_title="Enhanced LAB Color Analyzer", layout="wide", page_icon="🎨")
    st.title("Enhanced LAB Color Analyzer")
    st.markdown(
        """
        Welcome to the **Enhanced LAB Color Analyzer**!  
        Upload your **ISCC-NBS LAB colors** dataset and input your **LAB color** values to find the closest matching color with detailed visualizations.  
        Use the sidebar to upload your dataset and choose your input method.
        """
    )

    # Sidebar: Upload and Input Options
    st.sidebar.header("Upload & Input")
    uploaded_file = st.sidebar.file_uploader("Upload 'iscc_nbs_lab_colors.csv'", type=['csv'])
    linked_data_file = st.sidebar.file_uploader("Upload Getty AAT Linked Data JSON", type=['json'])
    
    with st.sidebar.expander("Instructions", expanded=False):
        st.markdown(
            """
            1. **Upload Dataset:** Upload the CSV file containing LAB values and color names.
            2. **Upload Linked Data (Optional):** Upload the JSON file with Getty AAT controlled vocabulary information.
            3. **Input LAB Color:** Choose your input method (sliders or manual entry).
            4. **Choose Delta-E Metric:** Select between Euclidean ΔE*76 or CIEDE2000.
            5. **Find Closest Color:** Click the button to see the closest matching color and various visualizations.
            """
        )

    # Let the user select the Delta-E metric
    delta_e_metric = st.sidebar.radio("Select Delta-E metric:", ("Euclidean ΔE76", "CIEDE2000"))
    if delta_e_metric == "Euclidean ΔE76":
        delta_e_func = calculate_delta_e_euclidean
    else:
        delta_e_func = calculate_delta_e_ciede2000

    if uploaded_file is None:
        st.info("Please upload your 'iscc_nbs_lab_colors.csv' file to begin.")
        debug_log("No dataset file uploaded.")
        return

    dataset_df = load_dataset(uploaded_file)
    if dataset_df.empty:
        debug_log("Dataset is empty after loading; terminating process.")
        return

    st.success("Dataset uploaded and validated successfully.")
    with st.expander("View Dataset Preview", expanded=False):
        st.dataframe(dataset_df.head())

    # Load Linked Data if available
    linked_data = {}
    if linked_data_file is not None:
        linked_data = load_linked_data(linked_data_file)

    # Input Method: Slider vs. Manual Entry
    input_method = st.sidebar.radio("Choose LAB Input Method:", ("Slider Input", "Manual Input"))
    if input_method == "Slider Input":
        st.sidebar.markdown("### LAB Color via Sliders")
        lab_l = st.sidebar.slider("L:", 0.0, 100.0, 50.0, 0.01)
        lab_a = st.sidebar.slider("A:", -128.0, 127.0, 0.0, 0.01)
        lab_b = st.sidebar.slider("B:", -128.0, 127.0, 0.0, 0.01)
    else:
        st.sidebar.markdown("### LAB Color via Manual Input")
        lab_l = st.sidebar.number_input("L (0-100):", min_value=0.0, max_value=100.0, value=50.0, step=0.1)
        lab_a = st.sidebar.number_input("A (-128 to 127):", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)
        lab_b = st.sidebar.number_input("B (-128 to 127):", min_value=-128.0, max_value=127.0, value=0.0, step=0.1)

    input_lab = [lab_l, lab_a, lab_b]

    if st.sidebar.button("Find Closest Color"):
        debug_log("User initiated color matching process.")
        if validate_lab_color(input_lab):
            with st.spinner("Processing..."):
                closest_color, delta_e = find_closest_color(input_lab, dataset_df, delta_e_func=delta_e_func)
                if closest_color is not None:
                    closest_color_name = closest_color['Color Name']
                    closest_lab = [closest_color['L'], closest_color['A'], closest_color['B']]
                    input_rgb = lab_to_rgb(input_lab)
                    closest_rgb = lab_to_rgb(closest_lab)

                    st.markdown("### **Results:**")
                    st.markdown(
                        f"""
                        **Input LAB Color:** L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}  
                        **Closest ISCC-NBS Color:** {closest_color_name}  
                        **Delta-E Value ({delta_e_metric}):** {delta_e:.2f}  
                        **Closest LAB Color:** L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}  
                        **Input RGB Color:** {input_rgb}  
                        **Closest RGB Color:** {closest_rgb}
                        """
                    )

                    # Organize visualizations into tabs
                    tabs = st.tabs([
                        "Color Comparison", "LAB Comparison", "3D LAB Plot", "Delta-E Histogram", "Color Density", "Pairwise Scatter"
                    ])
                    with tabs[0]:
                        fig1 = create_color_comparison_plot(input_rgb, closest_rgb, input_lab, closest_lab, closest_color_name, delta_e)
                        st.plotly_chart(fig1, use_container_width=True)
                    with tabs[1]:
                        fig2 = create_lab_comparison_bar(input_lab, closest_lab, closest_color_name, input_rgb, closest_rgb)
                        st.plotly_chart(fig2, use_container_width=True)
                    with tabs[2]:
                        fig3 = create_3d_lab_plot(input_lab, closest_lab, closest_color_name, dataset_df, input_rgb, closest_rgb)
                        st.plotly_chart(fig3, use_container_width=True)
                    with tabs[3]:
                        delta_e_values = delta_e_func(input_lab, dataset_df)
                        fig4 = create_delta_e_histogram(delta_e_values)
                        st.plotly_chart(fig4, use_container_width=True)
                    with tabs[4]:
                        fig5 = create_color_density_heatmap(dataset_df)
                        st.plotly_chart(fig5, use_container_width=True)
                    with tabs[5]:
                        fig_splom = create_pairwise_scatter_matrix(dataset_df, input_lab, closest_lab)
                        st.plotly_chart(fig_splom, use_container_width=True)

                    st.markdown("### **Results Table:**")
                    results = {
                        'Input LAB': f"L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}",
                        'Closest ISCC-NBS Color': closest_color_name,
                        'Delta-E Value': f"{delta_e:.2f}",
                        'Closest LAB': f"L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}",
                        'Input RGB': input_rgb,
                        'Closest RGB': closest_rgb
                    }
                    display_results_table(results)
                    debug_log("Color matching process completed successfully.")

                    # Offer an export option for the results table
                    csv_results = pd.DataFrame([results]).to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv_results,
                        file_name='color_match_results.csv',
                        mime='text/csv',
                    )
                else:
                    debug_log("Closest color not found; process terminated.")
        else:
            debug_log("Input LAB color validation failed.")

    # If Getty Linked Data is provided, display its semantic relationships visually.
    if linked_data:
        st.markdown("### **Semantic Relationships from Getty JSON:**")
        fig_sem = create_semantic_relationship_graph(linked_data)
        st.plotly_chart(fig_sem, use_container_width=True)

if __name__ == "__main__":
    main()
