import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from typing import Union, List, Tuple, Optional
from io import StringIO

# ------------------------------
# Utility Functions
# ------------------------------

def validate_lab_color(lab: Union[List[float], Tuple[float, float, float], np.ndarray]) -> bool:
    """
    Validate the input LAB color.

    Args:
        lab: A list, tuple, or array of three numerical values [L, A, B].

    Returns:
        True if the color is valid, otherwise False.
    """
    if not isinstance(lab, (list, tuple, np.ndarray)) or len(lab) != 3:
        st.error("Input LAB color must be a list, tuple, or array of three numerical values, nyah~!")
        return False

    try:
        L, A, B = float(lab[0]), float(lab[1]), float(lab[2])
    except (ValueError, TypeError):
        st.error("LAB components must be numerical values, meep!")
        return False

    if not (0 <= L <= 100):
        st.error("L component must be between 0 and 100, nyah~!")
        return False
    if not (-128 <= A <= 127):
        st.error("A component must be between -128 and 127, meep!")
        return False
    if not (-128 <= B <= 127):
        st.error("B component must be between -128 and 127, meep!")
        return False
    return True

@st.cache_data(show_spinner=False)
def lab_to_rgb(lab_color: Union[List[float], Tuple[float, float, float], np.ndarray]) -> Tuple[int, int, int]:
    """
    Convert a LAB color to its corresponding RGB color.

    Args:
        lab_color: A list/tuple/array of LAB values [L, A, B].

    Returns:
        A tuple (R, G, B) with values scaled to 0-255.
    """
    try:
        lab = LabColor(lab_l=lab_color[0], lab_a=lab_color[1], lab_b=lab_color[2])
        rgb = convert_color(lab, sRGBColor, target_illuminant='d65')
        # Clamp the RGB values between 0 and 1, then scale to 255
        rgb_clamped = (
            int(max(0, min(rgb.rgb_r, 1)) * 255),
            int(max(0, min(rgb.rgb_g, 1)) * 255),
            int(max(0, min(rgb.rgb_b, 1)) * 255)
        )
        return rgb_clamped
    except Exception as e:
        st.error(f"Error converting LAB to RGB: {e}")
        return (0, 0, 0)

def calculate_delta_e(input_lab: Union[List[float], Tuple[float, float, float]], dataset_df: pd.DataFrame) -> np.ndarray:
    """
    Compute the vectorized Delta-E (CIE76) values between the input LAB color and dataset LAB colors.

    Args:
        input_lab: A list or tuple of LAB values.
        dataset_df: DataFrame containing LAB columns.

    Returns:
        An array of Delta-E values.
    """
    input_lab_arr = np.array(input_lab)
    delta_e = np.linalg.norm(dataset_df[['L', 'A', 'B']].values - input_lab_arr, axis=1)
    return delta_e

def find_closest_color(input_lab: Union[List[float], Tuple[float, float, float]], 
                       dataset_df: pd.DataFrame) -> Tuple[Optional[pd.Series], Optional[float]]:
    """
    Find the closest color in the dataset to the input LAB color based on Delta-E.

    Args:
        input_lab: A list or tuple of LAB values.
        dataset_df: DataFrame with LAB columns.

    Returns:
        A tuple containing the closest color (as a pandas Series) and its Delta-E value.
    """
    delta_e_values = calculate_delta_e(input_lab, dataset_df)
    if np.all(np.isnan(delta_e_values)):
        st.error("Delta-E calculation resulted in all NaN values. Check your dataset and input LAB values, nyah~!")
        return None, None
    min_idx = np.nanargmin(delta_e_values)
    min_delta_e = delta_e_values[min_idx]
    closest_color = dataset_df.iloc[min_idx]
    return closest_color, min_delta_e

# ------------------------------
# Visualization Functions
# ------------------------------

def create_color_comparison_plot(input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int],
                                 input_lab: List[float], closest_lab: List[float],
                                 closest_color_name: str, delta_e: float) -> go.Figure:
    """
    Create a scatter plot comparing the input color and the closest matched color.
    """
    data = pd.DataFrame({
        'Color Type': ['Input Color', f'Closest: {closest_color_name}'],
        'RGB': [f'rgb{input_rgb}', f'rgb{closest_rgb}'],
        'LAB': [f'L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}',
                f'L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}'],
        'Delta-E': [delta_e, 'N/A']
    })

    fig = px.scatter(
        data_frame=data,
        x=[0, 1],
        y=[1, 1],
        color='RGB',
        hover_data=['Color Type', 'LAB', 'Delta-E'],
        labels={'x': '', 'y': ''},
        title='🖌️ Input Color vs Closest ISCC-NBS Color'
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
        x=0,
        y=1.05,
        text='Input Color',
        showarrow=False,
        font=dict(size=14, color='white'),
        xanchor='center'
    )
    fig.add_annotation(
        x=1,
        y=1.05,
        text=f'Closest: {closest_color_name}',
        showarrow=False,
        font=dict(size=14, color='white'),
        xanchor='center'
    )

    return fig

def create_lab_comparison_bar(input_lab: List[float], closest_lab: List[float],
                              closest_color_name: str, input_rgb: Tuple[int, int, int],
                              closest_rgb: Tuple[int, int, int]) -> go.Figure:
    """
    Create a bar plot comparing the LAB components of the input and closest colors.
    """
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
        title='🔍 LAB Value Comparison',
        template='plotly_dark',
        color_discrete_map=color_map
    )

    # Annotate differences for each LAB component
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

    return fig

def create_3d_lab_plot(input_lab: List[float], closest_lab: List[float],
                       closest_color_name: str, dataset_df: pd.DataFrame,
                       input_rgb: Tuple[int, int, int], closest_rgb: Tuple[int, int, int]) -> go.Figure:
    """
    Create a 3D scatter plot in LAB color space displaying the dataset colors,
    the input color, and its closest match.
    """
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
        title='🌐 3D LAB Color Space Visualization',
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
    return fig

def create_delta_e_histogram(delta_e_values: np.ndarray) -> go.Figure:
    """
    Create a histogram showing the distribution of Delta-E values.
    """
    fig = px.histogram(
        x=delta_e_values,
        nbins=30,
        title='📊 Delta-E Distribution',
        labels={'x': 'Delta-E Value', 'y': 'Count'},
        template='plotly_dark',
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
    Create a heatmap representing the density of colors in the A-B plane.
    """
    fig = px.density_heatmap(
        dataset_df,
        x='A',
        y='B',
        nbinsx=50,
        nbinsy=50,
        title='🔥 Color Density Heatmap in A-B Plane',
        labels={'A': 'A Component', 'B': 'B Component'},
        color_continuous_scale='Viridis',
        template='plotly_dark'
    )

    fig.update_layout(
        xaxis_title='A',
        yaxis_title='B',
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_pairwise_scatter_matrix(dataset_df: pd.DataFrame, input_lab: List[float],
                                     closest_lab: List[float]) -> go.Figure:
    """
    Create a scatter plot matrix to explore pairwise relationships between LAB components.
    """
    splom_df = dataset_df.copy()

    input_row = {'L': input_lab[0], 'A': input_lab[1], 'B': input_lab[2], 'Color Name': 'Input Color'}
    closest_row = {'L': closest_lab[0], 'A': closest_lab[1], 'B': closest_lab[2], 'Color Name': 'Closest Color'}
    splom_df = pd.concat([splom_df, pd.DataFrame([input_row, closest_row])], ignore_index=True)

    # Map each color name to its corresponding RGB value
    def map_color(color_name: str) -> str:
        if color_name == 'Input Color':
            return f'rgb{lab_to_rgb(input_lab)}'
        elif color_name == 'Closest Color':
            return f'rgb{lab_to_rgb(closest_lab)}'
        else:
            return 'lightgrey'
    
    splom_df['Color Group'] = splom_df['Color Name'].apply(map_color)

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
        title='🔍 Pairwise LAB Relationships',
        template='plotly_dark',
        dragmode='select',
        height=800
    )
    return fig_splom

def display_results_table(results: dict) -> None:
    """
    Display the results in a formatted HTML table.
    """
    df = pd.DataFrame([results])

    def color_rgb(cell: str) -> str:
        return f'<div style="background-color:{cell}; width:100px; height:20px;"></div>'

    # Apply HTML formatting to RGB columns
    df['Input RGB'] = df['Input RGB'].apply(lambda x: color_rgb(f'rgb{x[0]}, {x[1]}, {x[2]}'))
    df['Closest RGB'] = df['Closest RGB'].apply(lambda x: color_rgb(f'rgb{x[0]}, {x[1]}, {x[2]}'))

    st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# ------------------------------
# Data Loading Function
# ------------------------------

@st.cache_data(show_spinner=True)
def load_dataset(uploaded_file) -> pd.DataFrame:
    """
    Load and validate the dataset CSV file.

    Args:
        uploaded_file: The uploaded CSV file.

    Returns:
        A DataFrame with the dataset or an empty DataFrame if validation fails.
    """
    dataset_df = pd.read_csv(uploaded_file)
    required_columns = {'L', 'A', 'B', 'Color Name'}
    if not required_columns.issubset(set(dataset_df.columns)):
        missing = required_columns - set(dataset_df.columns)
        st.error(f"CSV file is missing required columns: {missing}")
        return pd.DataFrame()  # Return an empty DataFrame
    if dataset_df[list(required_columns)].isnull().any().any():
        st.error("CSV file contains missing values in required columns.")
        return pd.DataFrame()
    return dataset_df

# ------------------------------
# Main Application
# ------------------------------

def main() -> None:
    st.set_page_config(page_title="🎨 Enhanced LAB Color Analyzer", layout="wide", page_icon="🎨")
    st.title("🎨 Enhanced LAB Color Analyzer")
    st.markdown("""
    Welcome to the **Enhanced LAB Color Analyzer**! Upload your **ISCC-NBS LAB colors** dataset and input your **LAB color** values to find the closest matching color along with detailed visualizations.
    """)

    st.sidebar.header("🔧 Upload & Input")
    uploaded_file = st.sidebar.file_uploader("📂 Upload 'iscc_nbs_lab_colors.csv'", type=['csv'])

    if uploaded_file is None:
        st.info("📂 Please upload your 'iscc_nbs_lab_colors.csv' file to begin, nyah~!")
        return

    dataset_df = load_dataset(uploaded_file)
    if dataset_df.empty:
        return

    st.success("✅ Dataset uploaded and validated successfully.")
    st.write("📊 **Dataset Preview:**")
    st.dataframe(dataset_df.head())

    # LAB Input Sliders
    st.sidebar.markdown("### 🖌️ Input LAB Color")
    lab_l = st.sidebar.slider("L:", 0.0, 100.0, 50.0, 0.01)
    lab_a = st.sidebar.slider("A:", -128.0, 127.0, 0.0, 0.01)
    lab_b = st.sidebar.slider("B:", -128.0, 127.0, 0.0, 0.01)
    input_lab = [lab_l, lab_a, lab_b]

    if st.sidebar.button("🔍 Find Closest Color"):
        if validate_lab_color(input_lab):
            with st.spinner("Processing, nyah~..."):
                closest_color, delta_e = find_closest_color(input_lab, dataset_df)
                if closest_color is not None:
                    closest_color_name = closest_color['Color Name']
                    closest_lab = [closest_color['L'], closest_color['A'], closest_color['B']]

                    input_rgb = lab_to_rgb(input_lab)
                    closest_rgb = lab_to_rgb(closest_lab)

                    st.markdown("### 🟢 **Results:**")
                    st.write(f"**Input LAB Color:** L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}")
                    st.write(f"**Closest ISCC-NBS Color:** {closest_color_name}")
                    st.write(f"**Delta-E Value:** {delta_e:.2f}")
                    st.write(f"**Closest LAB Color:** L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}")
                    st.write(f"**Input RGB Color:** {input_rgb}")
                    st.write(f"**Closest RGB Color:** {closest_rgb}")

                    # Visualizations
                    st.markdown("## 📊 Visualizations")
                    col1, col2 = st.columns(2)
                    with col1:
                        fig1 = create_color_comparison_plot(input_rgb, closest_rgb, input_lab, closest_lab, closest_color_name, delta_e)
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        fig2 = create_lab_comparison_bar(input_lab, closest_lab, closest_color_name, input_rgb, closest_rgb)
                        st.plotly_chart(fig2, use_container_width=True)

                    col3, col4 = st.columns(2)
                    with col3:
                        fig3 = create_3d_lab_plot(input_lab, closest_lab, closest_color_name, dataset_df, input_rgb, closest_rgb)
                        st.plotly_chart(fig3, use_container_width=True)
                    with col4:
                        delta_e_values = calculate_delta_e(input_lab, dataset_df)
                        fig4 = create_delta_e_histogram(delta_e_values)
                        st.plotly_chart(fig4, use_container_width=True)

                    col5, col6 = st.columns(2)
                    with col5:
                        fig5 = create_color_density_heatmap(dataset_df)
                        st.plotly_chart(fig5, use_container_width=True)
                    with col6:
                        fig_splom = create_pairwise_scatter_matrix(dataset_df, input_lab, closest_lab)
                        st.plotly_chart(fig_splom, use_container_width=True)

                    # Results Table
                    st.markdown("### 📄 **Results Table:**")
                    results = {
                        'Input LAB': f"L={input_lab[0]}, A={input_lab[1]}, B={input_lab[2]}",
                        'Closest ISCC-NBS Color': closest_color_name,
                        'Delta-E': f"{delta_e:.2f}",
                        'Closest LAB': f"L={closest_lab[0]}, A={closest_lab[1]}, B={closest_lab[2]}",
                        'Input RGB': input_rgb,
                        'Closest RGB': closest_rgb
                    }
                    display_results_table(results)

if __name__ == "__main__":
    main()
